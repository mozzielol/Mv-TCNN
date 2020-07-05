import numpy as np
from configuration import conf
from models.model_definition import get_model, single_head_fully_connected


def converse_prediction(model_list, x, threshold, lamb):
    re_predict_index = None
    predictions = np.ones((x.shape[0])) * -1
    for i in range(len(model_list) - 1, -1, -1):
        if re_predict_index is None:
            pred = model_list[i].predict(x)
            diff = np.max(pred, axis=1)  # - np.min(pred, axis=1)
            probs = diff if lamb is None else diff / (diff + lamb - np.max(pred, axis=1) * lamb)
            re_predict_index = np.where(probs < (threshold[i][0] - 2 * threshold[i][1]))[0]
            predictions = np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1)
            predictions[re_predict_index] = -1
        else:
            pred = model_list[i].predict(x[re_predict_index])
            diff = np.max(pred, axis=1)  # - np.min(pred, axis=1)
            probs = diff if lamb is None else diff / (diff + lamb - np.max(pred, axis=1) * lamb)
            predictions[re_predict_index] = np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1)
            new_re_predict_index = np.where(probs < (threshold[i][0] - 2 * threshold[i][1]))[0]
            if i > 0:
                predictions[re_predict_index][new_re_predict_index] = -1
                re_predict_index = new_re_predict_index
        if len(re_predict_index) == 0:
            break

    return predictions


def plain_prediction(model_list, x, threshold, lamb):
    predictions = []
    probs = []
    for idx, model in enumerate(model_list):
        pred = model.predict(x)
        probs.append(np.max(pred[1], axis=1))
        predictions.append(np.argmax(pred[0], axis=1) + idx * 2 if conf.multi_head else np.argmax(pred[0], axis=1))
    probs = np.array(probs)
    predictions = np.array(predictions)
    predictions = predictions[np.argmax(probs, axis=0), np.arange(probs.shape[1])]
    return predictions


def train_model(data_loader, args):
    epochs = args.epochs
    verbose = args.verbose
    replay_rate = args.replay_rate
    select_sample = args.select_sample
    thresholds = []
    model_list = []
    initial_weights = None
    if args.same_initial:
        model = get_model(len(conf.task_labels[0]), args.num_centers)
        initial_weights = model.get_weights()

    for task_idx in range(conf.num_tasks):
        model = get_model(len(conf.task_labels[task_idx]), args.num_centers)
        if initial_weights is not None:
            model.set_weights(initial_weights)
        if task_idx == 0:
            x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
            model.fit(x, y, epochs=epochs, verbose=verbose)
        else:
            x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
            if replay_rate is not None:
                for i in range(task_idx):
                    x_, y_ = data_loader.sample(task_idx=i, whole_set=True)
                    replay_size = int(x_.shape[0] * replay_rate)  # // (task_idx + 1)
                    if select_sample:
                        probs = np.max(model_list[task_idx - i - 1].predict(x_), axis=1)
                        idx = np.argpartition(probs, replay_size // 2)
                        s_ = x_[idx[:replay_size // 2]]
                        idx = np.argpartition(probs, -replay_size // 2)
                        l_ = x_[idx[-replay_size // 2:]]
                        x_ = np.concatenate([s_, l_])
                        y_ = np.zeros_like(y_)[:x_.shape[0]]
                    else:
                        x_, y_ = x_[:replay_size], y_[:replay_size]
                        y_ = np.zeros_like(y_)
                    x = np.concatenate([x, x_])
                    y = np.concatenate([y, y_])
            model.fit(x, y, epochs=epochs, verbose=verbose)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x), axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))
        model_list.append(model)

    return model_list, thresholds
