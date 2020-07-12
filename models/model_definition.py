from models.layers import Probability_CLF_Mul
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dropout, BatchNormalization, \
    Activation, Concatenate, Embedding, multiply
from keras.models import Model
from configuration import conf


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


def get_model_multi_view_models(output_dim, num_centers, deep):
    if deep:
        return multi_view_model_conv(output_dim, num_centers)
    else:
        return multi_view_model(output_dim, num_centers)
