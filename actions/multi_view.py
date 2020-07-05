import numpy as np
from configuration import conf
from models.model_definition import get_model_multi_view_models
from keras.preprocessing.image import ImageDataGenerator


def rotate_grey_imgs(data, label, concat=True):
    if conf.dataset_name == 'mnist':
        img_shape = (data.shape[0], 28, 28, 1) if conf.is_conv else (data.shape[0], -1)
        data = data.reshape(-1, 28, 28)
    else:
        img_shape = (data.shape[0], 32, 32, 3) if conf.is_conv else (data.shape[0], -1)
        data = data.reshape(-1, 32, 32, 3)
    rotated_data = []
    rotated_label = [label]

    # Rotate
    for k in range(4):
        rotated_data.append(np.rot90(data, k=k, axes=(1, 2)).reshape(img_shape))
        rotated_label.append(label)
    rotated_data.append(data.reshape(img_shape))

    data = data.reshape(-1, 28, 28, 1) if conf.dataset_name == 'mnist' else data.reshape(-1, 32, 32, 3)

    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.fit(data)
    it = datagen.flow(data, label, batch_size=data.shape[0])
    agu_data, agu_label = it.next()
    rotated_data.append(agu_data.reshape(img_shape))
    rotated_label.append(agu_label)

    datagen = ImageDataGenerator(featurewise_std_normalization=True)
    datagen.fit(data)
    agu_data, agu_label = it.next()
    rotated_data.append(agu_data.reshape(img_shape))
    rotated_label.append(agu_label)

    datagen = ImageDataGenerator(zca_whitening=True)
    datagen.fit(data)
    it = datagen.flow(data, label, batch_size=data.shape[0])
    agu_data, agu_label = it.next()
    rotated_data.append(agu_data.reshape(img_shape))
    rotated_label.append(agu_label)
    # Shift
    datagen = ImageDataGenerator(width_shift_range=0.2)
    it = datagen.flow(data, label, batch_size=data.shape[0])
    agu_data, agu_label = it.next()
    rotated_data.append(agu_data.reshape(img_shape))
    rotated_label.append(agu_label)

    datagen = ImageDataGenerator(height_shift_range=0.2)
    it = datagen.flow(data, label, batch_size=data.shape[0])
    agu_data, agu_label = it.next()
    rotated_data.append(agu_data.reshape(img_shape))
    rotated_label.append(agu_label)

    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    it = datagen.flow(data, label, batch_size=data.shape[0])
    agu_data, agu_label = it.next()
    rotated_data.append(agu_data.reshape(img_shape))
    rotated_label.append(agu_label)

    if concat:
        return np.concatenate(rotated_data), np.concatenate(rotated_label)
    return rotated_data, rotated_label


def multi_view_train(data_loader, args):
    epochs = args.epochs
    verbose = args.verbose
    replay_rate = args.replay_rate
    select_sample = args.select_sample
    thresholds = []
    model_list = []
    initial_weights = None
    if args.same_initial:
        model = get_model_multi_view_models(len(conf.task_labels[0]) if conf.multi_head else 10, args.num_centers)
        initial_weights = model.get_weights()

    for task_idx in range(conf.num_tasks):
        model = get_model_multi_view_models(len(conf.task_labels[task_idx]) if conf.multi_head else 10, args.num_centers)
        if task_idx==0:
            print(model.summary())
        if initial_weights is not None:
            model.set_weights(initial_weights)
        x, y = data_loader.sample(task_idx=task_idx, whole_set=True)
        train_x, train_y = rotate_grey_imgs(x, y)
        task_out = np.ones(train_y.shape[0])
        # train_x.append(np.ones(x.shape[0]) * task_idx)
        model.fit(train_x, {'task_output': task_out, 'clf_output': train_y}, epochs=epochs, verbose=verbose)
        model_list.append(model)

    return model_list, thresholds

