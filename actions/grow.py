import numpy as np
from configuration import conf
from models.model_definition import get_model, multi_head_fully_connected, multi_head_grow
import keras.backend as K

class Grow_PNN():
    def __init__(self, data_loader, args):
        """
        :param data_loader:

        args -
            :param memory_size:
            :param epoch:
            :param num_centers
            :param threshold:
        """
        self.threshold = args.threshold
        self.memory_size = args.memory_size
        self.epoch = args.epoch
        self.data_loader = data_loader
        self.models = []
        self.stm = []
        self.prior = []
        self.num_samples = 0
        self.args = args

    def get_task_likelihood(self, x, posterior=False):
        max_idx = None
        max_tlh = 0
        for model_idx, model in enumerate(self.models):
            tlh = np.mean(model.predict(x)[1])
            if posterior:
                tlh *= self.prior[model_idx] / self.num_samples
            if tlh > max_tlh:
                max_tlh = tlh
                max_idx = model_idx
        return max_idx, max_tlh

    def store_stm(self, x, y):
        self.stm.append((x, y))
        if len(self.stm) > self.memory_size // conf.batch_size:
            self.add_expert()

    def add_expert(self):
        model = multi_head_fully_connected(10, 1)
        prior = 0
        mean_tlh = -1
        while mean_tlh < self.threshold:
            for data in self.stm:
                x, y = data
                task_out = np.ones(y.shape[0])
                model.train_on_batch(x, {'task_output': task_out, 'clf_output': y})
                if mean_tlh == -1:
                    self.num_samples += x.shape[0]
                    self.prior.append(x.shape[0])
            mean_tlh = 0
            for data in self.stm:
                x, y = data
                tlh = np.mean(model.predict(x)[1])
                mean_tlh += tlh
            mean_tlh /= len(self.stm)
        self.models.append(model)
        self.stm = []
        self.prior.append(prior)

    def train_expert(self, model_idx, x, y):
        task_out = np.ones(y.shape[0])
        self.num_samples += x.shape[0]
        self.prior[model_idx] += x.shape[0]
        return self.models[model_idx].train_on_batch(x, {'task_output': task_out, 'clf_output': y})

    def predict(self, x):
        model_idx, _ = self.get_task_likelihood(x, posterior=False)
        return  np.argmax(self.models[model_idx].predict(x)[0], axis=1) + model_idx * 2 \
            if conf.multi_head else np.argmax(self.models[model_idx].predict(x)[0], axis=1)

    def plain_predict(self, x):
        predictions = []
        probs = []
        for idx, model in enumerate(self.models):
            pred = model.predict(x)
            probs.append(np.max(pred[1], axis=1))
            predictions.append(np.argmax(pred[0], axis=1) + idx * 2 if conf.multi_head else np.argmax(pred[0], axis=1))
        probs = np.array(probs)
        predictions = np.array(predictions)
        predictions = predictions[np.argmax(probs, axis=0), np.arange(probs.shape[1])]
        return predictions, probs

    # def train(self):
    #     for task_idx in range(conf.num_tasks):
    #         print('Training Task %d' % task_idx)
    #         num_centers = 2
    #         x, y = self.data_loader.sample(task_idx=task_idx, whole_set=True)
    #         model = multi_head_fully_connected(2, num_centers)
    #         task_out = np.ones(y.shape[0])
    #         model.fit(x, {'task_output': task_out, 'clf_output': y}, verbose=self.args.verbose,
    #                   batch_size=conf.batch_size, epochs=self.epoch)
    #
    #         tlh = model.predict(x)[1].reshape(-1, ) < self.threshold
    #         while np.sum(tlh) > 0:
    #             print('Adding a new center', num_centers)
    #             num_centers += 1
    #             new_model = multi_head_fully_connected(2, num_centers, non_trainable=num_centers - 1)
    #             new_model.set_weights(model.get_weights())
    #             for layer in new_model.layers:
    #                 if layer.name.startswith('task_output'):
    #                     continue
    #                 layer.trainable = False
    #             new_model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
    #                               optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'}, )
    #             task_out = np.ones(len(y[tlh]))
    #             new_model.fit(x[tlh], {'task_output': task_out, 'clf_output': np.zeros_like(y[tlh])}, verbose=self.args.verbose,
    #                           batch_size=conf.batch_size, epochs=5)
    #             new_tlh = new_model.predict(x)[1].reshape(-1, ) < self.threshold
    #             if np.sum(new_tlh) >= np.sum(tlh):
    #                 break
    #             tlh = new_tlh
    #             model = new_model
    #
    #         self.models.append(model)
    #         print(np.mean(model.predict(x)[1]))

    def train(self):
        for task_idx in range(conf.num_tasks):
            print('Training Task %d' % task_idx)
            num_centers = 2
            x, y = self.data_loader.sample(task_idx=task_idx, whole_set=True)
            model = multi_head_grow(2, num_centers, non_trainable=0)
            try:
                model.set_weights(self.models[-1].get_weights())
                for layer in model.layers:
                    if layer.name.startswith('clf_output'):
                        continue
                    elif layer.name.startswith('task_output'):
                        continue
                        # if 'hidden' in layer.name:
                        #    continue
                        # pnn_weights = []
                        # pre_center = []
                        # for w_idx, weights in enumerate(self.models[-1].get_layer('task_output').weights):
                        #     pre_center.append(K.eval(weights))
                        # print(len(pre_center))
                        # pre_idx = 0
                        # for w_idx, weights in enumerate(layer.weights):
                        #     if 'out_of_distr' in weights.name:
                        #         pnn_weights.append(pre_center[pre_idx])
                        #         pre_idx += 1
                        #     else:
                        #         pnn_weights.append(pre_center[pre_idx])
                        # layer.set_weights(pnn_weights)
                        # continue
                    layer.trainable = False
                model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
                                   optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'},
                              loss_weights={'clf_output': 0.00001, 'task_output':0.99999} )
            except IndexError:
                pass
            task_out = np.ones(y.shape[0])
            model.fit(x, {'task_output': task_out, 'clf_output': y}, verbose=self.args.verbose,
                           batch_size=conf.batch_size, epochs=5)
            for t in range(conf.num_tasks):
                x, y = self.data_loader.sample(task_idx=t, whole_set=True)
                model = self.add_centers(x, y, model)
            self.models.append(model)
            for task_idx in range(conf.num_tasks):
                x, y = self.data_loader.sample(task_idx=task_idx, whole_set=True)
                print('Task ', task_idx, np.mean(model.predict(x)[1]))


    def add_centers(self, x, y, model):
        tlh = model.predict(x)[1].reshape(-1, ) > self.threshold
        num_centers = 2
        while np.sum(tlh) > 0:
            print('Adding a new center', num_centers)
            num_centers += 1
            new_model = multi_head_grow(2, num_centers, extra_node=num_centers - 1)
            new_model.set_weights(model.get_weights())
            for layer in new_model.layers:
                if layer.name.startswith('task_output'):
                    continue
                layer.trainable = False
            new_model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
                              optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'}, )
            task_out = np.zeros(len(y[tlh]))
            new_model.fit(x[tlh], {'task_output': task_out, 'clf_output': np.zeros_like(y[tlh])}, verbose=self.args.verbose,
                          batch_size=conf.batch_size, epochs=5)
            new_tlh = new_model.predict(x)[1].reshape(-1, ) > self.threshold
            # if np.sum(new_tlh) >= np.sum(tlh):
            #    break
            tlh = new_tlh
            model = new_model
        return model