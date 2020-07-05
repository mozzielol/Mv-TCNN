import numpy as np
from configuration import conf
from models.model_definition import get_model


def conditional_converse_prediction(model_list, x, threshold, lamb):
    re_predict_index = None
    predictions = np.ones((x.shape[0])) * -1
    for i in range(len(model_list) - 1, -1, -1):
        try:
            condition = 1 - np.mean(model_list[i + 1].predict(x), axis=1)
        except IndexError:
            condition = 1.0

        if re_predict_index is None:
            pred = model_list[i].predict(x)
            diff = np.max(pred, axis=1)  # - np.min(pred, axis=1)
            probs = diff if lamb is None else diff / (diff + lamb - np.max(pred, axis=1) * lamb)
            re_predict_index = np.where(probs * condition < (threshold[i][0] - 2 * threshold[i][1]))[0]
            predictions = np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1)
            predictions[re_predict_index] = -1
        else:
            pred = model_list[i].predict(x[re_predict_index])
            condition = condition[re_predict_index]
            diff = np.max(pred, axis=1)  # - np.min(pred, axis=1)
            probs = diff if lamb is None else diff / (diff + lamb - np.max(pred, axis=1) * lamb)
            predictions[re_predict_index] = np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1)
            new_re_predict_index = np.where(probs * condition < (threshold[i][0] - 2 * threshold[i][1]))[0]
            if i > 0:
                predictions[re_predict_index][new_re_predict_index] = -1
                re_predict_index = new_re_predict_index
        if len(re_predict_index) == 0:
            break

    return predictions


def conditional_prediction(model_list, x, threshold, lamb):
    prediction_idx = []
    prediction = []
    for i in range(len(model_list)):
        pred = model_list[i].predict(x)
        try:
            if i < len(model_list) - 1:
                condition = np.max(model_list[i+1].predict(x), axis=1)
            else:
                condition = 0.0
        except IndexError:
            condition = 0.0
        probs = (1 - condition) * np.max(pred,
                                         axis=1)  # * np.exp(-(np.max(pred, axis=1) - threshold[i][0])**2/(2*threshold[i][1]**2))
        prediction_idx.append(probs)
        prediction.append(np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1))

    idx = np.argmax(prediction_idx, axis=0)
    predictions = np.array(prediction)[idx, np.arange(idx.shape[0])]

    return predictions

def contion_by_others(model_list, x, threshold, lamb):
    prediction_idx = []
    prediction = []
    for i in range(len(model_list)):
        pred = model_list[i].predict(x)
        condition = np.ones_like(np.max(pred, axis=1))
        for j in range(i, len(model_list)):
            if i != j:
                condition *= (1 - np.max(model_list[j].predict(x), axis=1))
        # condition /= (len(model_list) - i)
        probs = (1 - condition) * np.max(pred,
                                         axis=1)  # * np.exp(-(np.max(pred, axis=1) - threshold[i][0])**2/(
        # 2*threshold[i][1]**2))
        probs = condition * np.max(pred, axis=1)
        prediction_idx.append(probs)
        prediction.append(np.argmax(pred, axis=1) + i * 2 if conf.multi_head else np.argmax(pred, axis=1))

    idx = np.argmax(prediction_idx, axis=0)
    predictions = np.array(prediction)[idx, np.arange(idx.shape[0])]

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
            x_, y_ = data_loader.sample(task_idx=task_idx - 1, whole_set=True)
            replay_size = int(x_.shape[0] * replay_rate)
            if select_sample:
                probs = np.max(model_list[task_idx - 1].predict(x_), axis=1)
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


def train_model_select(data_loader, args):
    epochs = args.epochs
    verbose = args.verbose
    replay_rate = args.replay_rate
    thresholds = []
    model_list = []

    for task_idx in range(conf.num_tasks):
        model = get_model(len(conf.task_labels[task_idx]), args.num_centers)
        if task_idx == 0:
            x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
            model.fit(x, y, epochs=epochs, verbose=verbose)
        else:
            x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
            x_, y_ = data_loader.sample(task_idx=task_idx - 1, whole_set=True)
            replay_size = int(x_.shape[0] * replay_rate)
            probs = np.max(model_list[-1].predict(x_), axis=1)
            idx = np.argpartition(probs, replay_size // 2)
            s_ = x_[idx[:replay_size // 2]]
            idx = np.argpartition(probs, -replay_size // 2)
            l_ = x_[idx[-replay_size // 2:]]
            x_ = np.concatenate([s_, l_])
            y_ = np.zeros_like(y_)[:x_.shape[0]]
            x = np.concatenate([x, x_])
            y = np.concatenate([y, y_])
            model.fit(x, y, epochs=epochs, verbose=verbose)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x), axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))
        model_list.append(model)

    return model_list, thresholds


def train_model_span(data_loader, args):
    epochs = args.epochs
    verbose = args.verbose
    replay_rate = args.replay_rate
    select_sample = args.select_sample
    span = args.memory_span
    thresholds = []
    model_list = []
    initial_weights = None
    memory = []
    if args.same_initial:
        model = get_model(len(conf.task_labels[0]), args.num_centers)
        initial_weights = model.get_weights()

    for task_idx in range(conf.num_tasks):
        model = get_model(len(conf.task_labels[task_idx]), args.num_centers)
        if initial_weights is not None:
            model.set_weights(initial_weights)

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        num_sample = x.shape[0]
        replay_size = int(x.shape[0] * replay_rate)
        memory.append((x[:replay_size], y[:replay_size]))

        for batch_idx in range(num_sample // conf.batch_size * epochs):
            x, y = data_loader.sample(task_idx=task_idx)
            for m_idx in range(len(memory) - 1):
                x_, y_ = memory[m_idx]
                y_ = np.zeros_like(y_)
                pre_index = np.random.choice(x_.shape[0], conf.batch_size)
                __x, __y = x_[pre_index], y_[pre_index]
                x = np.concatenate([x, __x])
                y = np.concatenate([y, __y])
            model.train_on_batch(x, y)
        model.fit(x, y, epochs=epochs, verbose=verbose)
        real_idx = 0
        for model_idx in range(task_idx - span, task_idx):
            if model_idx < 0:
                continue
            x, y = memory[real_idx]
            for m_idx in range(len(memory)):
                if m_idx == real_idx:
                    continue
                x_, y_ = memory[m_idx]
                y_ = np.zeros_like(y_)
                x = np.concatenate([x, x_])
                y = np.concatenate([y, y_])
            model_list[model_idx].fit(x, y, epochs=epochs//10, verbose=verbose)
            real_idx += 1
        if len(memory) > span:
            memory.pop(0)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model.predict(x), axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))
        model_list.append(model)
    return model_list, thresholds


def train_model_warmup(data_loader, args):
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
        model = get_model(len(conf.task_labels[0]), args.num_centers)
        if initial_weights is not None:
            model.set_weights(initial_weights)
        model_list.append(model)

    for task_idx in range(conf.num_tasks):

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        for model_idx in range(task_idx, len(model_list)):
            model = get_model(len(conf.task_labels[0]), args.num_centers)
            model.set_weights(model_list[model_idx].get_weights())
            if model_idx == task_idx:
                model.fit(x, y, epochs=epochs, verbose=verbose)
            else:
                model.fit(x, np.zeros_like(y), epochs=epochs, verbose=verbose)
            model_list[model_idx] = model

        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        pred = np.max(model_list[task_idx].predict(x), axis=1)
        thresholds.append((np.mean(pred), np.std(pred)))

    return model_list, thresholds