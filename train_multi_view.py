from configuration import conf
from utils.dataloader import Sequential_loader
import numpy as np
import argparse
import sys
import os
from actions.multi_view import rotate_grey_imgs
from utils.utils import mkdir


def parser_bool(data):
    if data == 'True':
        return True
    elif data == 'False':
        return False
    else:
        return data


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=parser_bool, default=False, help='verbose during training')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--replay_rate', type=float, default=0.2, help='Replay rate')
    parser.add_argument('--threshold', type=float, default=None, help='threshold for prediction')
    parser.add_argument('--save_to_file', type=str, default=None, help='the file to save the results')
    parser.add_argument('--num_centers', type=int, default=1, help='Number of centers in PNN')
    parser.add_argument('--lamb', type=int, default=None, help='lambda to calculate the prediction probability')
    parser.add_argument('--select_sample', type=parser_bool, default=True, help='select out samples during training')
    parser.add_argument('--conditional_train', type=parser_bool, default=True, help='whether train only prvious task')
    parser.add_argument('--conditional_prediction', type=str, default='converse_prediction', help='prediction action')
    parser.add_argument('--same_initial', type=parser_bool, default=True, help='Initialise the weights the same or not')
    parser.add_argument('--memory_span', type=int, default=None, help='Number of task to save in the memory')
    args = parser.parse_args(argv)
    return args


def train_prediction_action(args):
    from actions.multi_view import multi_view_train as train_model
    from actions.block import block_condition_by_others as prediction_action
    return train_model, prediction_action


def run(args):
    print(args)
    data_loader = Sequential_loader()
    train_model, prediction_action = train_prediction_action(args)
    threshold = args.threshold
    task_bias = len(conf.task_labels[0])
    print('Task bias', task_bias)
    # model_list, thresholds = train_model(data_loader, args)
    task_models = {}
    num_models = 1
    for i in range(num_models):
        model_list, thresholds = train_model(data_loader, args)
        task_models[i] = model_list

    for task_idx in range(conf.num_tasks):
        x, y = data_loader.sample(task_idx, dataset='test', whole_set=True)
        train_x, train_y = rotate_grey_imgs(x, y)
        # train_x.append(np.ones(x.shape[0]) * task_idx)
        pred = model_list[0].predict(train_x)
        print('Task ', task_idx)
        print(np.mean(pred[1]))
        print(np.mean(pred[1].reshape(-1) * np.max(pred[0], axis=1)))

    for task_idx in range(conf.num_tasks):
        x, y = data_loader.sample(task_idx, dataset='test', whole_set=True)
        pred = model_list[task_idx].predict(x)
        true_pred = np.argmax(y, axis=1) + task_idx * task_bias if conf.multi_head else np.argmax(y, axis=1)
        acc = np.sum(np.argmax(pred[0], axis=1) + task_idx * task_bias == true_pred) / true_pred.shape[0]
        print('Task ', task_idx)
        print(acc)


    avg_acc = 0
    for task_idx in range(conf.num_tasks):
        x, y = data_loader.sample(task_idx, dataset='test', whole_set=True)
        test_x, test_y = rotate_grey_imgs(x, y, concat=False)
        probabilities = []
        predictions = []
        for model_idx in range(conf.num_tasks):
            model_predictions = None
            model_probabilities = []
            prior = np.max(model_list[model_idx].predict(x)[0], axis=1)
            for data in test_x:
                pred = model_list[model_idx].predict(data)
                model_probabilities.append(pred[1].reshape(-1))
                if model_predictions is None:
                    model_predictions = pred[0]
                else:
                    model_predictions *= pred[0]
            probabilities.append(np.mean(model_probabilities, axis=0) * prior)
            if conf.multi_head:
                model_predictions = model_list[model_idx].predict(x)[0]
                predictions.append(np.argmax(model_predictions, axis=1) + model_idx * task_bias)
            else:
                model_predictions = model_list[model_idx].predict(x)[0]
                predictions.append(np.argmax(model_predictions, axis=1))
        probabilities = np.array(probabilities)
        predictions = np.array(predictions)
        predictions = predictions[np.argmax(probabilities, axis=0), np.arange(predictions.shape[1])]
        true_pred = np.argmax(y, axis=1) + task_idx * task_bias if conf.multi_head else np.argmax(y, axis=1)
        acc = np.sum(predictions == true_pred) / true_pred.shape[0]
        avg_acc += acc
        print('Task ', task_idx, acc)
        print('Right selection : ', np.sum(np.argmax(probabilities, axis=0) == task_idx) / probabilities.shape[1])
    print(avg_acc / conf.num_tasks)

    # path = './ckpt/' + conf.dataset_name
    # mkdir(path)
    # for m_idx, m in enumerate(model_list):
    #     filename = path + '/task%d.pkl'%m_idx
    #     m.save(filename)




"""
Results: 
    dataset - num_centers - epochs - accuracy
    CIFAR10 -     4 -          5 -   0.4662
    CIFAR10 -     2 -          10 -   0.4662
    CIFAR10 -     2 -          10 -   0.4993
CIFAR 10:
Task  0 0.657
Right selection :  0.6735
Task  1 0.223
Right selection :  0.2465
Task  2 0.323
Right selection :  0.352
Task  3 0.7645
Right selection :  0.7815
Task  4 0.529
Right selection :  0.5405
0.4993
"""
def run_combinations(args):
    import itertools
    params = {
        'replay_rate': [None],  # np.linspace(0.05, 0.3, 6),
        'num_centers': [5],  # np.arange(1, 6),
        'lamb': [None],
        'select_sample': [False],
        'conditional_train': [True],
        'conditional_prediction': [True],
        'same_initial': [False],
        'memory_span': [None],  # [1,2,3,4],
        'deep':[False],

    }
    flat = [[(k, v) for v in vs] for k, vs in params.items()]
    combinations = [dict(items) for items in itertools.product(*flat)]
    for c_idx, c in enumerate(combinations):  # 36 for mnist
        print('Test model %d/%d' % (c_idx + 1, len(combinations)))
        for key in c:
            setattr(args, key, c[key])
        run(args)


if __name__ == '__main__':
    # different output nodes
    args = get_args(sys.argv[1:])
    setattr(args, 'save_to_file', 'block_pred.csv')
    run_combinations(args)
