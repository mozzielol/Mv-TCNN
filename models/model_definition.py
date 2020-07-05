from models.layers import Vertical_space_layer, Gaussian_Likelihood_Layer, Probability_CLF_Mul, PNN_Growth
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dropout, BatchNormalization, Activation, Concatenate, Embedding, multiply
from keras.models import Model
from configuration import conf
from keras import regularizers
from keras.constraints import unit_norm
# from keras.applications import VGG16, ResNet50V2
from keras.layers import GlobalAveragePooling2D

def fully_connected(output_dim, num_centers):
    inputs = Input(shape=(784,))
    archi = Dense(200, activation='relu')(inputs)
    archi = Dense(200, activation='relu')(archi)
    task_id = Probability_CLF_Mul(output_dim, num_centers=num_centers, name='task_output')(archi)
    model = Model(inputs=inputs, outputs=task_id)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'], )

    return model


def multi_head_fully_connected(output_dim, num_centers, non_trainable=0):
    inputs = Input(shape=(784,))
    archi = Dense(200)(inputs)
    # archi = BatchNormalization()(archi)
    archi = Activation('relu')(archi)
    # archi = Dropout(0.5)(archi)
    archi = Dense(200)(archi)
    # archi = BatchNormalization()(archi)
    archi = Activation('relu')(archi)
    # archi = Dropout(0.5)(archi)
    # archi = Dense(200)(archi)
    # archi = Activation('relu')(archi)
    # archi = Dropout(0.25)(archi)
    clf = Dense(output_dim, activation='softmax', name='clf_output')(archi)
    task_id = Probability_CLF_Mul(1, num_centers=num_centers, non_trainable=non_trainable, name='task_output')(archi)
    model = Model(inputs=inputs, outputs=[clf, task_id])
    model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
                  optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'}, )

    return model


def multi_head_grow(output_dim, num_centers, non_trainable=0, extra_node=0):
    inputs = Input(shape=(784,))
    archi = Dense(200, activation='relu')(inputs)
    # archi = LeakyReLU(alpha=0.2)(archi)
    archi = Dense(200, activation='relu')(archi)
    # task_hidden = Dense(200, activation='selu', name='task_output_hidden')(archi)
    clf_hidden = Dense(200, activation='relu', name='clf_output_hidden')(archi)
    clf_hidden = Dense(200, activation='relu', name='clf_output_hidden2')(clf_hidden)
    # archi = LeakyReLU(alpha=0.2)(archi)
    clf = Dense(output_dim, activation='softmax', name='clf_output')(clf_hidden)
    task_id = PNN_Growth(1, num_centers=num_centers, non_trainable=non_trainable, extra_node=extra_node,
                         name='task_output')(archi)
    model = Model(inputs=inputs, outputs=[clf, task_id])
    model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
                  optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'}, )

    return model


def convolutional_nn(output_dim, num_centers):
    inputs = Input(shape=(32, 32, 3))
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Flatten()(archi)
    archi = Dense(200, activation='relu')(archi)
    archi = Dense(200, activation='relu')(archi)
    archi = Dense(400, activation='relu')(archi)
    task_id = Gaussian_Likelihood_Layer(output_dim, num_centers=num_centers, name='task_output')(archi)
    model = Model(inputs=inputs, outputs=task_id)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['mse'], )
    return model


def convolutional_nn_multi_head(output_dim, num_centers):
    inputs = Input(shape=(28, 28, 1)) if conf.dataset_name == 'mnist' else Input(shape=(32, 32, 3))
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(64, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(128, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(128, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Flatten()(archi)
    # model_input = multiply([inputs[i], label_embedding])
    archi = Dense(4096, activation='relu')(archi)
    archi = Dense(2048, activation='relu')(archi)
    archi = Dense(1024, activation='relu')(archi)
    # inputs.append(label)
    clf = Dense(output_dim, activation='softmax', name='clf_output')(archi)
    task_id = Probability_CLF_Mul(1, num_centers=num_centers, non_trainable=0, name='task_output')(archi)
    model = Model(inputs=inputs, outputs=[clf, task_id])
    model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
                  optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'},
                  loss_weights={'clf_output': 0.9, 'task_output': 0.1})
    return model


def mnist_cnn(output_dim):
    # label = Input(shape=(1,), dtype='int32')
    # label_embedding = Flatten()(Embedding(conf.num_tasks, 784)(label))
    inputs = Input(shape=(28, 28, 1)) if conf.dataset_name == 'mnist' else Input(shape=(32, 32, 3))
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Flatten()(archi)
    # model_input = multiply([inputs[i], label_embedding])
    archi = Dense(400, activation='relu')(archi)
    archi = Dense(400, activation='relu')(archi)
    # inputs.append(label)
    clf = Dense(output_dim, activation='softmax', name='clf_output')(archi)
    model = Model(inputs=inputs, outputs=clf)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )

    return model

def cifar_cnn(output_dim):
    # label = Input(shape=(1,), dtype='int32')
    # label_embedding = Flatten()(Embedding(conf.num_tasks, 784)(label))
    inputs = Input(shape=(28, 28, 1)) if conf.dataset_name == 'mnist' else Input(shape=(32, 32, 3))
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(64, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(128, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Flatten()(archi)
    # model_input = multiply([inputs[i], label_embedding])
    archi = Dense(1024, activation='relu')(archi)
    archi = Dense(512, activation='relu')(archi)
    archi = Dense(400, activation='relu')(archi)
    # inputs.append(label)
    clf = Dense(output_dim, activation='softmax', name='clf_output')(archi)
    model = Model(inputs=inputs, outputs=clf)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )

    return model



def multi_view_model(output_dim, num_centers, non_trainable=0):
    # label = Input(shape=(1,), dtype='int32')
    # label_embedding = Flatten()(Embedding(conf.num_tasks, 784)(label))
    inputs = Input(shape=(28, 28, 1)) if conf.dataset_name == 'mnist' else Input(shape=(32, 32, 3))
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Flatten()(archi)
    # model_input = multiply([inputs[i], label_embedding])
    archi = Dense(400, activation='relu')(archi)
    archi = Dense(400, activation='relu')(archi)
    # inputs.append(label)
    clf = Dense(output_dim, activation='softmax', name='clf_output')(archi)
    task_id = Probability_CLF_Mul(1, num_centers=num_centers, non_trainable=non_trainable, name='task_output')(archi)
    model = Model(inputs=inputs, outputs=[clf, task_id])
    model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
                  optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'},
                  loss_weights={'clf_output': 0.9, 'task_output': 0.1})

    return model


def multi_view_model_conv(output_dim, num_centers, non_trainable=0):
    # label = Input(shape=(1,), dtype='int32')
    # label_embedding = Flatten()(Embedding(conf.num_tasks, 784)(label))
    inputs = Input(shape=(28, 28, 1)) if conf.dataset_name == 'mnist' else Input(shape=(32, 32, 3))
    archi = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(64, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Conv2D(128, (3, 3), padding='same', activation='relu')(archi)
    archi = MaxPooling2D(pool_size=(2, 2))(archi)
    archi = Flatten()(archi)
    # model_input = multiply([inputs[i], label_embedding])
    archi = Dense(1024, activation='relu')(archi)
    archi = Dense(512, activation='relu')(archi)
    archi = Dense(400, activation='relu')(archi)
    # inputs.append(label)
    clf = Dense(output_dim, activation='softmax', name='clf_output')(archi)
    task_id = Probability_CLF_Mul(1, num_centers=num_centers, non_trainable=non_trainable, name='task_output')(archi)
    model = Model(inputs=inputs, outputs=[clf, task_id])
    model.compile(loss={'clf_output': 'categorical_crossentropy', 'task_output': 'binary_crossentropy'},
                  optimizer='adam', metrics={'clf_output': 'accuracy', 'task_output': 'mse'},
                  loss_weights={'clf_output': 0.9, 'task_output': 0.1})

    return model



def get_model(output_dim, num_centers):
    if conf.dataset_name == 'mnist':
        return fully_connected(output_dim, num_centers)
    else:
        return convolutional_nn(output_dim, num_centers)


def get_model_multi_head(output_dim, num_centers):
    if conf.dataset_name == 'mnist':
        return multi_view_model(output_dim, num_centers)
    else:
        return convolutional_nn_multi_head(output_dim, num_centers)

def get_model_multi_view_models(output_dim, num_centers, deep):
    if deep:
        return convolutional_nn_multi_head(output_dim, num_centers)
    else:
        return multi_view_model_conv(output_dim, num_centers)
