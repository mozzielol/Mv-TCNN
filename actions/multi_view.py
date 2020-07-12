from keras.preprocessing.image import ImageDataGenerator
from configuration import conf
import numpy as np


class Multi_view:
    def __init__(self):
        self.datagen = [
            ImageDataGenerator(rotation_range=270),
            ImageDataGenerator(featurewise_center=True),
            ImageDataGenerator(featurewise_std_normalization=True),
            ImageDataGenerator(zca_whitening=True, zca_epsilon=0.1),
            ImageDataGenerator(width_shift_range=0.2),
            ImageDataGenerator(height_shift_range=0.2),
            ImageDataGenerator(horizontal_flip=True, vertical_flip=True),
            ImageDataGenerator(zoom_range=0.15),
            ImageDataGenerator(shear_range=0.15), ]

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

    def augment(self, x, y=None, concat=False, num_runs=10):
        augment_data = [x]
        augment_label = [y]
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
