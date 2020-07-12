from keras.preprocessing.image import ImageDataGenerator
from configuration import conf
import numpy as np


def rotate_270(data):
    if conf.dataset_name == 'mnist':
        img_shape = (data.shape[0], 28, 28, 1) if conf.is_conv else (data.shape[0], -1)
        data = data.reshape(-1, 28, 28)
    elif conf.dataset_name == 'timh':
        img_shape = (data.shape[0], 28, 28, 1) if conf.is_conv else (data.shape[0], -1)
        data = data.reshape(-1, 28, 28)
    else:
        img_shape = (data.shape[0], 32, 32, 3) if conf.is_conv else (data.shape[0], -1)
        data = data.reshape(-1, 32, 32, 3)
    return np.rot90(data, k=3, axes=(1, 2)).reshape(img_shape)


class Multi_view:
    def __init__(self):
        self.datagen = [
            ImageDataGenerator(samplewise_center=True),
            ImageDataGenerator(samplewise_std_normalization=True),
            ImageDataGenerator(featurewise_center=True),
            ImageDataGenerator(featurewise_std_normalization=True),
            ImageDataGenerator(zca_whitening=True, zca_epsilon=0.1),
            ImageDataGenerator(zca_whitening=True),
            ImageDataGenerator(rotation_range=180),
            ImageDataGenerator(width_shift_range=0.4),
            ImageDataGenerator(height_shift_range=0.4),
            ImageDataGenerator(horizontal_flip=True),
            ImageDataGenerator(vertical_flip=True),
            ImageDataGenerator(zoom_range=0.3),
            ImageDataGenerator(shear_range=30), ]


    def fit(self, x):
        for gen in self.datagen:
            gen.fit(x)

    def flow(self, x, y):
        augment_data = []
        augment_label = []
        for gen in self.datagen:
            data, label = gen.flow(x, y, batch_size=conf.batch_size).next()
            augment_data.append(data)
            augment_label.append(label)

    def augment(self, x, y=None, concat=False, num_runs=1):
        augment_data = [x, rotate_270(x)]
        augment_label = [y, y]
        if y is None:
            for _ in np.arange(num_runs):
                for gen in self.datagen:
                    data = gen.flow(x, batch_size=x.shape[0]).next()
                    augment_data.append(data)
            if concat:
                return np.concatenate(augment_data)
            return augment_data
        for _ in np.arange(num_runs):
            for gen in self.datagen:
                data, label = gen.flow(x, y, batch_size=x.shape[0]).next()
                augment_data.append(data)
                augment_label.append(label)
        if concat:
            return np.concatenate(augment_data), np.concatenate(augment_label)
        return augment_data, augment_label

    def augment_test_data(self, x, num_runs=10):
        augment_data = [x, rotate_270(x)]
        y = np.arange(x.shape[0])
        for idx, gen in enumerate(self.datagen):
            for _ in np.arange(num_runs):
                data, label = gen.flow(x, y, batch_size=x.shape[0]).next()
                augment_data.append(data[label.argsort()])
        return augment_data

