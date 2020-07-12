import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


# Multiple Centers
class Probability_CLF_Mul(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=2, non_trainable=0, **kwargs):
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        super(Probability_CLF_Mul, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = {}

        for idx in range(self.output_dim):
            self.centers[idx] = []

            if idx in range(self.non_trainable):
                trainable = False
            else:
                trainable = True
            for c in range(self.num_centers):
                W = self.add_weight(name='center%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
                                    trainable=True)
                self.centers[idx].append(W)
        super(Probability_CLF_Mul, self).build(input_shape)

    def call(self, x, training=None):
        logits = []
        re_logits = []
        # Fixed Sigma
        sigma = 5.
        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c]), sigma))

            G = tf.stack(G, axis=1)
            P = tf.reduce_sum(G, axis=1) / (
                        tf.reduce_sum(G, axis=1) + self.num_centers - tf.reduce_max(G, axis=1) * self.num_centers)
            # P = K.max(G,axis=1)
            logits.append(P)

            P = tf.reduce_sum(G, axis=1) / self.num_centers
            re_logits.append(P)

        logits = tf.stack(logits, axis=1)
        re_logits = tf.stack(re_logits, axis=1)

        if training in {0, False}:
            return logits

        return K.in_train_phase(logits,
                                logits,
                                training=training)

    def gaussian_activation(self, x, sigma=None):
        sigma = 0.05 if sigma == None else sigma
        return tf.exp(-tf.reduce_sum(x, axis=1) / (2. * sigma * sigma))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
