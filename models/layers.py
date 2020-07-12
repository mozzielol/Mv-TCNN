import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras import regularizers, constraints


class Probability_CLF_Mul(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=2, non_trainable=0, num_out_distr=0, out_non_trainable=0,
                 kernel_regularizer=None, kernel_constraint=None, activation=None, **kwargs):
        self.centers = {}
        self.out_centers = {}
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        self.activation = activation
        self.num_out_distr = num_out_distr
        self.out_nun_trainable = out_non_trainable
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        super(Probability_CLF_Mul, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.kernels = []

        for idx in range(self.output_dim):
            self.centers[idx] = []
            self.out_centers[idx] = []
            if idx in range(self.non_trainable):
                trainable = False
            else:
                trainable = True
            for c in range(self.num_centers):
                W = self.add_weight(name='center%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
                                    trainable=trainable, regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
                self.centers[idx].append(W)

            if idx in range(self.out_nun_trainable):
                trainable = False
            else:
                trainable = True
            # for c in range(self.num_out_distr):
            #     W = self.add_weight(name='out_center%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
            #                         trainable=trainable)
            #     self.out_centers[idx].append(W)

        super(Probability_CLF_Mul, self).build(input_shape)

    def call(self, x, training=None):
        logits = []
        re_logits = []
        sigma = 1.
        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c]), sigma))

            G = tf.stack(G, axis=1)

            P = tf.reduce_sum(G, axis=1) / (
                    tf.reduce_sum(G, axis=1) + self.num_centers - tf.reduce_max(G, axis=1) * self.num_centers)
            # diff = self.num_centers * tf.reduce_max(G, axis=1) - tf.reduce_sum(G, axis=1)
            # P = tf.reduce_max(G, axis=1)
            logits.append(P)

            P = tf.reduce_sum(G, axis=1) / self.num_centers
            re_logits.append(P)

        logits = tf.stack(logits, axis=1)
        re_logits = tf.stack(re_logits, axis=1)

        if self.activation is not None:
            logits = self.activation(logits)
        if training in {0, False}:
            return logits

        return K.in_train_phase(logits,
                                logits,
                                training=training)

    def gaussian_activation(self, x, sigma=None):
        sigma = 1.
        return tf.exp(-tf.reduce_sum(x, axis=1) / (2. * sigma * sigma))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
