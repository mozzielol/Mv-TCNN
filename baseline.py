from configuration import conf
from utils.dataloader import Sequential_loader
import numpy as np
import argparse
import sys
import os
from actions.multi_view import rotate_grey_imgs
from utils.utils import mkdir
from models.model_definition import mnist_cnn, cifar_cnn


EPOCHS = 10
task_bias = len(conf.task_labels[0])

data_loader = Sequential_loader()
model_list = []
for task_idx in range(conf.num_tasks):
    x, y = data_loader.sample(task_idx, whole_set=True)
    model = mnist_cnn(len(conf.task_labels[task_idx]))
    model.fit(x, y, epochs=EPOCHS)
    model_list.append(model)


avg_acc = 0
for task_idx in range(conf.num_tasks):
    x, y = data_loader.sample(task_idx, dataset='test', whole_set=True)
    predictions = []
    probabilities = []
    for model_idx, model in enumerate(model_list):
        pred = model.predict(x)
        probabilities.append(np.max(pred, axis=1))
        est_pred = np.argmax(pred, axis=1) + model_idx * task_bias if conf.multi_head else np.argmax(pred, axis=1)
        predictions.append(est_pred)
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    predictions = predictions[np.argmax(probabilities, axis=0), np.arange(predictions.shape[1])]
    true_pred = np.argmax(y, axis=1) + task_idx * task_bias if conf.multi_head else np.argmax(y, axis=1)
    acc = np.sum(predictions == true_pred) / true_pred.shape[0]
    avg_acc += acc

print('='*100)
print(conf.dataset_name, task_bias, avg_acc / conf.num_tasks)