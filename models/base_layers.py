import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


# Multiple Centers
class Add_forward(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=1, **kwargs):
        self.output_dim = output_dim
        self.num_centers = num_centers
        super(Add_forward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = {}

        for idx in range(self.output_dim):
            self.centers[idx] = []
            for c in range(self.num_centers):
                w_mean = self.add_weight(name='w_mean%d_%d' % (idx, c), shape=(input_shape[1],),
                                         initializer=keras.initializers.Constant(),
                                         trainable=False)
                w_rho = self.add_weight(name='w_sigma%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
                                        trainable=True)
                w_sigma = tf.log(1.0 + tf.exp(w_rho))
                self.centers[idx].append([w_mean, w_sigma])
        super(Add_forward, self).build(input_shape)

    def call(self, x):
        logits = []
        probs = []
        for idx in range(self.output_dim):
            G = []
            for c in range(self.num_centers):
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c][0]),
                                                  self.centers[idx][c][1]))
                logits.append(tf.reduce_sum(
                    tf.squared_difference(x, self.centers[idx][c][0]) / 2 * self.centers[idx][c][1] *
                    self.centers[idx][c][1],
                    axis=1))

            G = tf.stack(G, axis=1)
            P = (tf.reduce_max(G, axis=1) * (self.num_centers + 1) - tf.reduce_sum(G, axis=1)) / self.num_centers
            probs.append(P)

        probs = tf.stack(probs, axis=1)
        logits = tf.stack(logits, axis=1)
        logits = self.custom_relu(logits, probs)

        return logits

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def gaussian_activation(self, x, sigma=None):
        return tf.exp(-tf.reduce_sum(x / (2. * sigma * sigma), axis=1))

    def custom_relu(self, x, probs):
        isPositive = K.cast(K.greater_equal(probs, 0.0), K.floatx())
        return x * isPositive
