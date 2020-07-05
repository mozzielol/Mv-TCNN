from configuration import conf
from utils.dataloader import Sequential_loader
import numpy as np
import argparse
import sys
from models.model_definition import get_model, multi_head_fully_connected
from utils.utils import one_hot
import keras


def generate_imgs(generator, task_idx, num_samples):
    noise = np.random.normal(0, 1, (num_samples, 100))
    sampled_labels = np.random.randint(task_idx * 2, task_idx * 2 + 2, size=(num_samples,))
    gen_imgs = generator.predict([noise, sampled_labels]).reshape(num_samples, -1)
    gen_imgs = 0.5 * gen_imgs + 0.5
    labels = sampled_labels - task_idx * 2 if conf.multi_head else sampled_labels
    return gen_imgs, labels


def generate_imgs_by_labels(generator, label, num_samples):
    noise = np.random.normal(0, 1, (num_samples, 100))
    sampled_labels = np.ones(num_samples, ) * label
    gen_imgs = generator.predict([noise, sampled_labels]).reshape(num_samples, -1)
    gen_imgs = 0.5 * gen_imgs + 0.5
    return gen_imgs, sampled_labels.astype(np.int64)


def block_condition_by_others(model_list, x, threshold, lamb):
    prediction_idx = []
    prediction = []
    for i in range(len(model_list)):
        pred = model_list[i].predict(x)[0]
        condition = np.ones_like(np.max(pred, axis=1))
        for j in range(i, len(model_list)):
            if i != j:
                condition *= (1 - np.max(model_list[j].predict(x)[1], axis=1))
        # 2*threshold[i][1]**2))
        probs = condition * np.max(model_list[i].predict(x)[1], axis=1)
        prediction_idx.append(probs)
        prediction.append(np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1))

    idx = np.argmax(prediction_idx, axis=0)
    predictions = np.array(prediction)[idx, np.arange(idx.shape[0])]

    return predictions

def train_model(data_loader, args):
    # different output nodes
    T = args.gans_threshold
    epochs = args.epochs
    verbose = args.verbose
    threshold = args.threshold
    thresholds = []
    model_list = []
    generator = keras.models.load_model('pnn.pkl')
    for task_idx in range(conf.num_tasks):
        print('Training model %d' % task_idx)
        model = get_model(len(conf.task_labels[task_idx]), args.num_centers)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        num_samples = x.shape[0]
        for gen_idx in range(task_idx):
            x_, y_ = generate_imgs(generator, gen_idx, num_samples * 10)
            sample_idx = model_list[gen_idx].predict(x_)
            right_predictions = np.argmax(sample_idx, axis=1) == y_
            x_ = x_[right_predictions]
            y_ = y_[right_predictions]
            sample_idx = sample_idx[right_predictions]
            y_ = one_hot(y_, 2)
            sample_idx = np.max(sample_idx, axis=1) > T
            x_ = x_[sample_idx]
            y_ = y_[sample_idx]
            x_ = x_[:int(num_samples // task_idx)]
            y_ = y_[:int(num_samples // task_idx)]
            x = np.concatenate([x, x_])
            y = np.concatenate([y, np.zeros_like(y_)])
            print('Task %d add %d/%d samples' % (gen_idx, x_.shape[0], num_samples))

        model.fit(x, y, epochs=epochs, verbose=verbose)

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x), axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))

        model_list.append(model)

    threshold = [threshold] * conf.num_tasks if threshold is not None else thresholds
    return model_list, threshold
    #
    # avg_acc = 0
    # for task_idx in range(conf.num_tasks):
    #     x, y = data_loader.sample(task_idx=task_idx, dataset='test', whole_set=True)
    #     pred = converse_prediction(x)
    #     acc = np.sum(pred == (np.argmax(y, axis=1) + task_idx * 2)) / y.shape[0]
    #     print('task {} accuracy : {:.4f}'.format(task_idx, acc))
    #     avg_acc += acc
    # print(avg_acc / conf.num_tasks)
    # for task_idx in range(conf.num_tasks):
    #     x, y = data_loader.sample(task_idx=task_idx, dataset='test', whole_set=True)
    #     print(task_idx, '****')
    #     for m in model_list:
    #         print(np.mean(np.max(m.predict(x), axis=1)))


def train_model_multi_head(data_loader, args):
    from models.model_definition import multi_head_fully_connected
    T = args.gans_threshold
    epochs = args.epochs
    verbose = args.verbose
    replay_rate = args.replay_rate
    select_sample = args.select_sample
    threshold = args.threshold
    thresholds = []
    model_list = []
    generator = keras.models.load_model('pnn_gans.pkl')
    for task_idx in range(conf.num_tasks):
        print('Training model %d' % task_idx)
        task_len = task_idx * 2 + 2
        model = multi_head_fully_connected(task_len, args.num_centers)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        y = np.argmax(y, axis=1)
        num_samples = x.shape[0]
        task_y = np.ones(y.shape[0])
        for gen_idx in range(task_idx):
            x_, y_ = generate_imgs(generator, gen_idx, num_samples * 10)
            sample_idx = model_list[gen_idx].predict(x_)
            right_predictions = np.argmax(sample_idx[0], axis=1) == y_
            x_ = x_[right_predictions]
            y_ = y_[right_predictions]
            sample_idx = sample_idx[1][right_predictions]
            sample_idx = np.max(sample_idx, axis=1) > thresholds[gen_idx][0]
            x_ = x_[sample_idx]
            y_ = y_[sample_idx]
            x_ = x_[:int(num_samples // task_idx)]
            y_ = y_[:int(num_samples // task_idx)]
            x = np.concatenate([x, x_])
            task_y = np.concatenate([task_y, np.zeros(y_.shape[0])])
            y = np.concatenate([y, y_])
            print('Task %d add %d/%d samples' % (gen_idx, x_.shape[0], num_samples // task_idx))

        for i in range(task_idx):
            x_, y_ = data_loader.sample(task_idx=i, whole_set=True)
            replay_size = int(x_.shape[0] * replay_rate)  # // (task_idx + 1)
            if select_sample:
                probs = np.max(model_list[task_idx - i - 1].predict(x_)[1], axis=1)
                idx = np.argpartition(probs, replay_size // 2)
                s_ = x_[idx[:replay_size // 2]]
                idx = np.argpartition(probs, -replay_size // 2)
                l_ = x_[idx[-replay_size // 2:]]
                x_ = np.concatenate([s_, l_])
                y_ = y_[:x_.shape[0]]
                task_out_ = np.zeros(y_.shape[0])[:x_.shape[0]]
            else:
                x_, y_ = x_[:replay_size], y_[:replay_size]
                task_out_ = np.zeros(y_.shape[0])
            y_ = np.argmax(y_, axis=1)
            x = np.concatenate([x, x_])
            y = np.concatenate([y, y_])
            task_y = np.concatenate([task_y, task_out_])

        y = one_hot(y, task_len)
        model.fit(x, {'task_output': task_y, 'clf_output': y}, epochs=epochs, verbose=verbose)

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x)[1], axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))

        model_list.append(model)

    threshold = [threshold] * conf.num_tasks if threshold is not None else thresholds
    return model_list, threshold


def train_model_multi_head_sequential(data_loader, args):
    from models.model_definition import multi_head_fully_connected
    T = args.gans_threshold
    epochs = args.epochs
    verbose = args.verbose
    threshold = args.threshold
    thresholds = []
    model_list = []
    generator = keras.models.load_model('pnn_gans.pkl')
    for task_idx in range(conf.num_tasks):
        print('Training model %d' % task_idx)
        task_len = task_idx * 2 + 2
        model = multi_head_fully_connected(task_len, args.num_centers)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        y = np.argmax(y, axis=1)
        num_samples = x.shape[0]
        for gen_label in range(np.min(y)):
            x_, y_ = generate_imgs_by_labels(generator, gen_label, num_samples // 2)
            sample_idx = model_list[-1].predict(x_)
            right_predictions = np.argmax(sample_idx[0], axis=1) == y_
            x_ = x_[right_predictions]
            y_ = y_[right_predictions]
            sample_idx = sample_idx[1][right_predictions]
            sample_idx = np.max(sample_idx, axis=1) > T
            x = np.concatenate([x, x_[sample_idx]])
            y = np.concatenate([y, y_[sample_idx]])
            print('Image %d add %d/%d samples' % (gen_label, np.sum(sample_idx), num_samples))
        y = one_hot(y, task_len)
        model.fit(x, [y, np.ones(y.shape[0])], epochs=epochs, verbose=verbose)

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x)[1], axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))

        model_list.append(model)

    threshold = [threshold] * conf.num_tasks if threshold is not None else thresholds
    return model_list, threshold


def train_model_sequential(data_loader, args):
    # different output nodes
    T = args.gans_threshold
    epochs = args.epochs
    verbose = args.verbose
    threshold = args.threshold
    thresholds = []
    model_list = []
    generator = keras.models.load_model('./gans/pnn_gans.pkl')
    for task_idx in range(conf.num_tasks):
        print('Training model %d' % task_idx)
        task_len = task_idx * 2 + 2
        model = get_model(task_len, args.num_centers)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        y = np.argmax(y, axis=1)
        num_samples = x.shape[0]
        for gen_label in range(np.min(y)):
            x_, y_ = generate_imgs_by_labels(generator, gen_label, num_samples // 2)
            sample_idx = model_list[-1].predict(x_)
            right_predictions = np.argmax(sample_idx, axis=1) == y_
            x_ = x_[right_predictions]
            y_ = y_[right_predictions]
            sample_idx = sample_idx[right_predictions]
            sample_idx = np.max(sample_idx, axis=1) > T
            x = np.concatenate([x, x_[sample_idx]])
            y = np.concatenate([y, y_[sample_idx]])
            print('Image %d add %d/%d samples' % (gen_label, np.sum(sample_idx), num_samples))
        y = one_hot(y, task_len)
        model.fit(x, y, epochs=epochs, verbose=verbose)

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x), axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))

        model_list.append(model)

    threshold = [threshold] * conf.num_tasks if threshold is not None else thresholds
    return model_list, threshold
