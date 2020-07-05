import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras import regularizers, constraints

# Multiple Centers
class Gaussian_Likelihood_Layer(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=1, non_trainable=0, **kwargs):
        self.centers = {}
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        super(Gaussian_Likelihood_Layer, self).__init__(**kwargs)

    def build(self, input_shape):

        for idx in range(self.output_dim):
            self.centers[idx] = []
            for c in range(self.num_centers):
                w_mean = self.add_weight(name='w_mean%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
                                         trainable=True)
                w_rho = self.add_weight(name='w_sigma%d_%d' % (idx, c), shape=(1,), initializer='uniform',
                                        trainable=True)
                w_sigma = tf.log(1.0 + tf.exp(w_rho))
                self.centers[idx].append([w_mean, w_sigma])
        super(Gaussian_Likelihood_Layer, self).build(input_shape)

    def call(self, x, training=None):
        logits = []
        re_logits = []
        # Fixed Sigma

        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c][0]),
                                                  self.centers[idx][c][1]))

            G = tf.stack(G, axis=1)
            P = (tf.reduce_max(G, axis=1) * (self.num_centers + 1) - tf.reduce_sum(G, axis=1)) / self.num_centers
            logits.append(P)

            P = tf.reduce_sum(G, axis=1) / self.num_centers
            re_logits.append(P)

        logits = tf.stack(logits, axis=1)
        re_logits = tf.reduce_sum(logits, axis=1) / (
                tf.reduce_sum(logits, axis=1) + self.output_dim - tf.reduce_max(logits, axis=1) * self.output_dim)

        if training in {0, False}:
            return logits

        return K.in_train_phase(logits,
                                logits,
                                training=training)

    def gaussian_activation(self, x, sigma):
        return tf.exp(-tf.reduce_sum(x / (2. * sigma * sigma), axis=1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class Vertical_space_layer(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=1, extra_nodes=1, **kwargs):
        self.centers = {}
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.extra_nodes = extra_nodes
        super(Vertical_space_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        for idx in range(self.output_dim):
            self.centers[idx] = []
            for c in range(self.num_centers + self.extra_nodes):
                w_mean = self.add_weight(name='w_mean%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
                                         trainable=True)
                w_rho = self.add_weight(name='w_sigma%d_%d' % (idx, c), shape=(1,), initializer='uniform',
                                        trainable=True)
                w_sigma = tf.log(1.0 + tf.exp(w_rho))
                self.centers[idx].append([w_mean, w_sigma])
        super(Vertical_space_layer, self).build(input_shape)

    def call(self, x, training=None):
        logits = []
        for idx in range(self.output_dim):
            G = []
            for c in range(self.num_centers):
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c][0]),
                                                  self.centers[idx][c][1]))
            N = []
            for c in range(self.num_centers, self.num_centers + self.extra_nodes):
                N.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c][0]),
                                                  self.centers[idx][c][1]))
            G = tf.stack(G, axis=1)
            N = tf.stack(N, axis=1)
            P = (tf.reduce_max(G, axis=1) * (self.num_centers + 1) - tf.reduce_sum(G, axis=1)) / self.num_centers
            neg_P = tf.reduce_sum(N, axis=1) / self.extra_nodes
            logits.append(P - neg_P)
        logits = tf.stack(logits, axis=1)
        if training in {0, False}:
            return logits

        return K.in_train_phase(logits,
                                logits,
                                training=training)

    def gaussian_activation(self, x, sigma):
        return tf.exp(-tf.reduce_sum(x / (2. * sigma * sigma), axis=1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class Probability_CLF_Mul(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=2, non_trainable=0, num_out_distr=0, out_non_trainable=0,
                 kernel_regularizer=None, kernel_constraint=None, activation=None, **kwargs):
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
        self.centers = {}
        self.out_centers = {}
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
                                    trainable=trainable, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
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


class PNN_Growth(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=2, non_trainable=0, extra_node=0, activation=None, **kwargs):
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        self.activation = activation
        self.extra_node = extra_node
        super(PNN_Growth, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = {}
        self.out_centers = {}
        # self.task_weights = self.add_weight(name='task_weights', shape=(input_shape[1], input_shape[1]), initializer='uniform',
        #                             trainable=True)
        # self.task_bias = self.add_weight(name='task_bias', shape=(input_shape[1],), initializer='uniform',
        #                            trainable=True)
        for idx in range(self.output_dim):
            self.centers[idx] = []
            self.out_centers[idx] = []

            for c in range(self.num_centers):
                W = self.add_weight(name='center%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
                                    trainable=True if c > self.extra_node else False)
                self.centers[idx].append(W)

            for c in range(self.non_trainable):
                W = self.add_weight(name='out_of_distr%d_%d' % (idx, c), shape=(input_shape[1],), initializer='uniform',
                                    trainable=False)
                self.out_centers[idx].append(W)

        super(PNN_Growth, self).build(input_shape)

    def call(self, x, training=None):
        logits = []
        sigma = 1.
        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c]), sigma))

            Neg = []
            for c in range(self.non_trainable):
                G.append(self.gaussian_activation(tf.squared_difference(x, self.out_centers[idx][c]), sigma))

            G = tf.stack(G, axis=1)
            try:
                Neg = tf.stack(Neg, axis=1)
            except IndexError:
                pass

            P = tf.reduce_sum(G, axis=1) / (
                   tf.reduce_sum(G, axis=1) + self.num_centers - tf.reduce_max(G, axis=1) * self.num_centers + tf.reduce_sum(Neg))
            # P = tf.reduce_max(G, axis=1)
            logits.append(P)
        logits = tf.stack(logits, axis=1)
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


class Task_prob_with_pre(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=1, non_trainable=0, **kwargs):
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        super(Task_prob_with_pre, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = {}
        # self.kernels = []
        for idx in range(self.output_dim):
            self.centers[idx] = []

            if idx in range(self.non_trainable):
                trainable = False
            else:
                trainable = True
            for c in range(self.num_centers):
                W = self.add_weight(name='center%d_%d' % (idx, c), shape=(input_shape[0][1],), initializer='uniform',
                                    trainable=True)
                self.centers[idx].append(W)

        super(Task_prob_with_pre, self).build(input_shape)

    def call(self, x):
        x, pre_outputs = x

        logits = self.get_logits(x)
        diff = self.get_logits(pre_outputs)

        return [logits, diff]

    def gaussian_activation(self, x, sigma=None):
        sigma = 0.05 if sigma == None else sigma
        return tf.exp(-tf.reduce_sum(x, axis=1) / (2. * sigma * sigma))

    def get_logits(self, x, sigma=1.):
        logits = []
        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):
                # G += self.gaussian_activation(x - self.centers[idx][c])
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c]), sigma))

            G = tf.stack(G, axis=1)

            P = tf.reduce_sum(G, axis=1) / (
                    tf.reduce_sum(G, axis=1) + self.num_centers - tf.reduce_max(G, axis=1) * self.num_centers)

            logits.append(P)

            P = tf.reduce_sum(G, axis=1) / self.num_centers

        logits = tf.stack(logits, axis=1)
        return logits

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]


class PNN_multiple_outputs(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=2, non_trainable=0, **kwargs):
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        super(PNN_multiple_outputs, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = {}
        # self.kernels = []

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

        super(PNN_multiple_outputs, self).build(input_shape)

    def call(self, x, training=None):
        logits = []
        re_logits = []
        # Fixed Sigma
        sigma = 1.
        # W = keras.activations.softmax(self.kernel)
        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):
                # G += self.gaussian_activation(x - self.centers[idx][c])
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c]), sigma))

            G = tf.stack(G, axis=1)

            P = tf.reduce_sum(G, axis=1) / (
                    tf.reduce_sum(G, axis=1) + self.num_centers - tf.reduce_max(G, axis=1) * self.num_centers)

            logits.append(P)

            P = tf.reduce_sum(G, axis=1) / self.num_centers
            re_logits.append(P)

        logits = tf.stack(logits, axis=1)
        re_logits = tf.reduce_sum(logits, axis=1, keep_dims=False) / (
                tf.reduce_sum(logits, axis=1, keep_dims=False) + self.output_dim - tf.reduce_max(logits, axis=1,
                                                                                                 keep_dims=False) * self.output_dim)

        return logits

    def gaussian_activation(self, x, sigma=None):
        sigma = 0.05 if sigma == None else sigma
        return tf.exp(-tf.reduce_sum(x, axis=1) / (2. * sigma * sigma))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class PNN_poseterior(Layer):
    """docstring for Probability_CLF"""

    def __init__(self, output_dim, num_centers=2, non_trainable=0, activation=None, **kwargs):
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.non_trainable = non_trainable
        self.activation = activation
        super(PNN_poseterior, self).__init__(**kwargs)

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

		# Arguments
		    args (tensor): mean and log of variance of Q(z|X)

		# Returns
		    z (tensor): sampled latent vector
		"""

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        res = z_mean + K.exp(0.5 * z_log_var) * epsilon
        return res

    def build(self, input_shape):
        self.centers = {}
        # self.kernels = []

        for idx in range(self.output_dim):
            self.centers[idx] = []

            if idx in range(self.non_trainable):
                trainable = False
            else:
                trainable = True
            for c in range(self.num_centers):
                mean = self.add_weight(name='center%d_%d' % (idx, c), shape=(input_shape[0][1],),
                                       initializer='random_normal', trainable=True)
                self.centers[idx].append(mean)

        super(PNN_poseterior, self).build(input_shape)

    def call(self, x, training=None):
        x, sigma = x
        sigma = tf.log(1.0 + tf.exp(sigma))
        logits = []
        re_logits = []
        # Fixed Sigma
        # W = keras.activations.softmax(self.kernel)
        for idx in range(self.output_dim):

            G = []
            for c in range(self.num_centers):
                # G += self.gaussian_activation(x - self.centers[idx][c])
                G.append(self.gaussian_activation(tf.squared_difference(x, self.centers[idx][c]), sigma))

            G = tf.stack(G, axis=1)

            P = tf.reduce_sum(G, axis=1) / (
                    tf.reduce_sum(G, axis=1) + self.num_centers - tf.reduce_max(G, axis=1) * self.num_centers)

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
        return tf.exp(-tf.reduce_sum(x / (2. * sigma * sigma), axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
