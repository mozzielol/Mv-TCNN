import numpy as np
from configuration import conf
from models.model_definition import get_model, multi_head_fully_connected


class DPM():
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
        self.task_models = {}

    def get_task_likelihood(self, x, models, posterior=False):
        max_idx = None
        max_tlh = 0
        for model_idx, model in enumerate(models):
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
        model = multi_head_fully_connected(10, 2)
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
        max_tlh = 0
        max_task = 0
        model_idx = 0
        for i in self.task_models.keys():
            model_idx, tlh = self.get_task_likelihood(x, self.task_models[i], posterior=True)
            if tlh > max_tlh:
                max_tlh = tlh
                model_idx = model_idx
                max_task = i
        return np.argmax(self.task_models[max_task][model_idx].predict(x)[0], axis=1)

    def train(self):
        for task_idx in range(conf.num_tasks):
            x, y = self.data_loader.sample(task_idx=task_idx, whole_set=True)
            num_samples = x.shape[0]
            print('Training Task %d' % task_idx)
            for iteration in range(num_samples // conf.batch_size * self.epoch):
                x, y = self.data_loader.sample(task_idx=task_idx)
                model_idx, tlh = self.get_task_likelihood(x, self.models)
                print(iteration, '/', num_samples // conf.batch_size * self.epoch, tlh, model_idx)
                if tlh > self.threshold and model_idx is not None:
                    loss = self.train_expert(model_idx, x, y)
                    print('Num Experts: {}, training model {}, TLH: {}, loss: {}'.format(len(self.models), model_idx, tlh,
                                                                                       loss))
                else:
                    self.store_stm(x, y)
                    print('store sample to the STM, STM length = {}, num experts = {}'.format(len(self.stm), len(self.models)))
            self.task_models[task_idx] = self.models
            self.models = []
            for task_idx in range(conf.num_tasks):
                x, y = self.data_loader.sample(task_idx=task_idx, dataset='test', whole_set=True)
                pred = self.predict(x)
                true_pred = np.argmax(y, axis=1) + task_idx * 2 if conf.multi_head else np.argmax(y, axis=1)
                acc = np.sum(pred == true_pred) / y.shape[0]
                print('task {} accuracy : {:.4f}'.format(task_idx, acc))


def dpm_train(data_loader, args):
    epochs = args.epochs
    verbose = args.verbose
    replay_rate = args.replay_rate
    select_sample = args.select_sample
    thresholds = []
    model_list = []
    initial_weights = None
    if args.same_initial:
        model = get_model(10, args.num_centers)
        initial_weights = model.get_weights()

    for task_idx in range(conf.num_tasks):
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        num_samples = x.shape[0]
        for iteration in num_samples // conf.batch_size * epochs:
            x, y = data_loader.sample(task_idx=task_idx)

        model = get_model(10, args.num_centers)
        if initial_weights is not None:
            model.set_weights(initial_weights)
        model_list.append(model)

    for task_idx in range(conf.num_tasks):
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        num_sample = x.shape[0]
        for batch_idx in range(num_sample // conf.batch_size * epochs):
            x, y = data_loader.sample(task_idx=task_idx, batch_size=conf.batch_size)

            # which model should be trained
            confidences = []
            for model_idx in range(len(model_list)):
                pred = model_list[model_idx].predict(x)
                confidences.append(np.mean(np.max(pred, axis=1)))

            model_list[np.argmax(confidences)].train_on_batch(x, y)
        thresholds.append((0.9, 0))

    return model_list, thresholds


def dpm_prediction(model_list, x, threshold, lamb):
    predictions = []
    for data in x:
        max_probs = 0.
        max_pred = None
        for model in model_list:
            pred = model.predict(data.reshape(1, -1))
            if np.max(pred) > max_probs:
                max_probs = np.max(pred)
                max_pred = np.argmax(pred)
        predictions.append(max_pred)
    return np.array(predictions)
