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
    parser.add_argument('--save_to_file', type=str, default=None, help='the file to save the results')
    parser.add_argument('--num_centers', type=int, default=2, help='Number of centers in PNN')
    parser.add_argument('--same_initial', type=parser_bool, default=True, help='Initialise the weights the same or not')
    args = parser.parse_args(argv)
    return args


def run(args):
    data_loader = Sequential_loader()
    print(args)
    model_list, multi_view_functions = train_model(data_loader, args)

    avg_acc = 0
    for task_idx in range(conf.num_tasks):
        x, y = data_loader.sample(task_idx, dataset='test', whole_set=True)
        probabilities = []
        predictions = []
        for model_idx in range(conf.num_tasks):
            test_x = multi_view_functions[model_idx].augment(x, concat=False)
            model_probabilities = []
            prior = np.max(model_list[model_idx].predict(x)[0], axis=1)
            for data in test_x:
                pred = model_list[model_idx].predict(data)
                model_probabilities.append(np.max(pred[0], axis=1) * pred[1].reshape(-1))
            probabilities.append(np.mean(model_probabilities, axis=0) * prior)
            if conf.multi_head:
                model_predictions = model_list[model_idx].predict(x)[0]
                predictions.append(np.argmax(model_predictions, axis=1) + model_idx * 2)
            else:
                model_predictions = model_list[model_idx].predict(x)[0]
                predictions.append(np.argmax(model_predictions, axis=1))
        probabilities = np.array(probabilities)
        predictions = np.array(predictions)
        predictions = predictions[np.argmax(probabilities, axis=0), np.arange(predictions.shape[1])]
        true_pred = np.argmax(y, axis=1) + task_idx * 2 if conf.multi_head else np.argmax(y, axis=1)
        acc = np.sum(predictions == true_pred) / true_pred.shape[0]
        avg_acc += acc
        print('Task ', task_idx, acc)
        print('Right selection : ', np.sum(np.argmax(probabilities, axis=0) == task_idx) / probabilities.shape[1])
    print(avg_acc / conf.num_tasks)

    path = './ckpt/' + conf.dataset_name
    mkdir(path)
    for m_idx, m in enumerate(model_list):
        filename = path + '/task%d.pkl' % m_idx
        m.save(filename)


"""
Results: 
    dataset - num_centers - epochs - accuracy
    CIFAR10 -     4 -          5 -   0.4662
    CIFAR10 -     2 -          10 -   0.4662
    CIFAR10 -     2 -          10 -   0.4993
                                        0.5660
                                        
CIFAR10: multi_view_model_conv, zca_epsilon = default, 0.5855
CIFAR100: multi_view_model_conv, zca_epsilon=0.1, 0.2814

    CIFAR100 - 5 - 30 - 0.281
Task  0 0.5965
Right selection :  0.6005
Task  1 0.179
Right selection :  0.189
Task  2 0.5795
Right selection :  0.609
Task  3 0.739
Right selection :  0.749
Task  4 0.736
Right selection :  0.755
0.5660000000000001


mnist
Task  1 0.9030362389813908
Right selection :  0.9045053868756121
Task  2 0.9887940234791889
Right selection :  0.9887940234791889
Task  3 0.9315206445115811
Right selection :  0.93202416918429
Task  4 0.9687342410489158
Right selection :  0.9697428139183056
0.9563366513536243
"""

if __name__ == '__main__':
    # different output nodes
    args = get_args(sys.argv[1:])
    setattr(args, 'save_to_file', 'block_pred.csv')
    run(args)
