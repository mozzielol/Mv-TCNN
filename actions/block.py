import numpy as np
from configuration import conf
from models.model_definition import get_model_multi_head


def block_likelihood(res, keep_dim=False, block_size=10):
    extra_index = res.shape[0] % block_size
    extra_values = res[-extra_index:]
    resize_values = res[:-extra_index]
    tlh = resize_values.reshape(-1, block_size)
    tlh_std = np.std(tlh, axis=1, keepdims=True)
    tlh_mean = np.mean(tlh, axis=1, keepdims=True)
    if keep_dim:
        tlh_mean = np.repeat(tlh_mean, block_size, axis=1).reshape(-1, )
        extra_mean = np.repeat(np.mean(extra_values), len(extra_values))
        return np.append(tlh_mean, extra_mean)
    else:
        extra_mean = np.mean(extra_values)
        extra_std = np.std(extra_values)
        tlh_mean = tlh_mean.reshape(-1, )
        tlh_std = tlh_std.reshape(-1, )
        return np.append(tlh_mean, extra_mean), np.append(tlh_std, extra_std)


def block_prediction(model_list, x, block_size, lamb):
    prediction = []
    res = []

    for i in range(len(model_list)):
        pred = model_list[i].predict(x)
        res.append(block_likelihood(np.max(pred[1], axis=1), keep_dim=True, block_size=block_size))
        # res.append(np.max(pred[1], axis=1) * np.max(pred[0], axis=1))
        prediction.append(np.argmax(pred[0], axis=1) + i * 2)
    res = np.array(res)
    prediction = np.array(prediction)
    return prediction[np.argmax(res, axis=0), np.arange(prediction.shape[1])]


def block_converse_prediction(model_list, x, threshold, lamb):
    re_predict_index = None
    predictions = np.ones((x.shape[0])) * -1
    for i in range(len(model_list) - 1, -1, -1):
        if re_predict_index is None:
            pred = model_list[i].predict(x)[1]
            diff = np.max(pred, axis=1)  # - np.min(pred, axis=1)
            probs = diff if lamb is None else diff / (diff + lamb - np.max(pred, axis=1) * lamb)
            re_predict_index = np.where(probs < (threshold[i][0] - 2 * threshold[i][1]))[0]
            predictions = np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1)
            predictions[re_predict_index] = -1
        else:
            pred = model_list[i].predict(x[re_predict_index])[1]
            diff = np.max(pred, axis=1)  # - np.min(pred, axis=1)
            probs = diff if lamb is None else diff / (diff + lamb - np.max(pred, axis=1) * lamb)
            predictions[re_predict_index] = np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred,
                                                                                                              axis=1)
            new_re_predict_index = np.where(probs < (threshold[i][0] - 2 * threshold[i][1]))[0]
            if i > 0:
                predictions[re_predict_index][new_re_predict_index] = -1
                re_predict_index = new_re_predict_index
        if len(re_predict_index) == 0:
            break

    return predictions


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


def block_train(data_loader, args):
    epochs = args.epochs
    verbose = args.verbose
    replay_rate = args.replay_rate
    select_sample = args.select_sample
    thresholds = []
    model_list = []
    initial_weights = None
    if args.same_initial:
        model = get_model_multi_head(len(conf.task_labels[0]) if conf.multi_head else 10, args.num_centers)
        initial_weights = model.get_weights()

    for task_idx in range(conf.num_tasks):
        model = get_model_multi_head(len(conf.task_labels[task_idx]) if conf.multi_head else 10, args.num_centers)
        if initial_weights is not None:
            model.set_weights(initial_weights)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        task_out = np.ones(y.shape[0])
        if replay_rate is not None:
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
                x = np.concatenate([x, x_])
                y = np.concatenate([y, y_])
                task_out = np.concatenate([task_out, task_out_])
        model.fit(x, {'task_output': task_out, 'clf_output': y}, epochs=epochs, verbose=verbose)

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x)[1], axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))
        model_list.append(model)

    return model_list, thresholds
