from configuration import conf
from utils.dataloader import Sequential_loader
import numpy as np
import argparse
import sys
import os
from utils.utils import mkdir
from actions.train_action import multi_view_train as train_model


def parser_bool(data):
    if data == 'True':
        return True
    elif data == 'False':
        return False
    else:
        return data


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=parser_bool, default=True, help='verbose during training')
    parser.add_argument('--deep', type=parser_bool, default=False, help='deeper model')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--num_centers', type=int, default=2, help='Number of centers in PNN')
    parser.add_argument('--same_initial', type=parser_bool, default=False, help='Initialise the weights the same or not')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs to augment test data')
    args = parser.parse_args(argv)
    return args


def run(args):
    data_loader = Sequential_loader()
    print(args)
    print('#'*100, flush=True)
    print('Dataset: ', conf.dataset_name, flush=True)
    print('num of task: ', conf.num_tasks, flush=True)
    print('model: ', args.deep, flush=True)
    print('num of centers: ', args.num_centers, flush=True)
    print('#' * 100, flush=True)
    model_list, multi_view_functions = train_model(data_loader, args, False)
    task_bias = len(conf.task_labels[0])
    avg_acc = 0
    for task_idx in range(conf.num_tasks):
        x, y = data_loader.sample(task_idx, dataset='test', whole_set=True)
        probabilities = []
        predictions = []
        for model_idx in range(conf.num_tasks):
            test_x = multi_view_functions[model_idx].augment_test_data(x, num_runs=args.num_runs)
            model_probabilities = []
            prior = np.max(model_list[model_idx].predict(x)[0], axis=1)
            model_predictions = np.zeros_like(model_list[model_idx].predict(x)[0])
            for data in test_x:
                pred = model_list[model_idx].predict(data)
                model_probabilities.append(np.max(pred[0], axis=1) * pred[1].reshape(-1))
                model_predictions += pred[0]

            probabilities.append(np.mean(model_probabilities, axis=0))
            predictions.append(np.argmax(model_predictions, axis=1) + model_idx * task_bias)

        probabilities = np.array(probabilities)
        predictions = np.array(predictions)
        predictions = predictions[np.argmax(probabilities, axis=0), np.arange(predictions.shape[1])]
        true_pred = np.argmax(y, axis=1) + task_idx * task_bias if conf.multi_head else np.argmax(y, axis=1)
        acc = np.sum(predictions == true_pred) / true_pred.shape[0]
        avg_acc += acc
        print('Task ', task_idx, acc, flush=True)
        print('Right selection : ', np.sum(np.argmax(probabilities, axis=0) == task_idx) / probabilities.shape[1], flush=True)
    print(avg_acc / conf.num_tasks, flush=True)


    # path = './ckpt/' + conf.dataset_name
    # mkdir(path)
    # for m_idx, m in enumerate(model_list):
    #     filename = path + '/task%d.pkl' % m_idx
    #     m.save(filename)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    run(args)
