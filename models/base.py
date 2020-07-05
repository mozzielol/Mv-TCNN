from abc import abstractmethod
import tensorflow as tf
from configuration import conf


class Base_model(object):
	"""docstring for Base_model"""
	def __init__(self, X_holder, y_holder):
		self.X_holder = X_holder
		self.y_holder = y_holder

		self.initialize()
		self.build()
		self.inference()
		self.summary()


	@abstractmethod
	def build(self):
		pass

	@abstractmethod
	def inference(self):
		pass

	def summary(self):
		try:
			print(self.model.summary())
		except AttributeError:
			self.get_model_info()

		with tf.name_scope('summaries'):
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('accuracy', self.accuracy)
			tf.summary.histogram('histogram accuracy', self.accuracy)
			self.summary_op = tf.summary.merge_all()


	def reset(self):
		return tf.global_variables_initializer()

	def fully_reset(self):
		self.net_params = {}
		return tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

	def initialize(self):
		self.net_params = {}
		with tf.variable_scope('Hyper_params', reuse=tf.AUTO_REUSE):
			self.gstep = tf.get_variable('global_step',trainable=False,initializer=tf.constant(0),dtype=tf.int32)
			self.lam = tf.get_variable('lambda',trainable=False,initializer=tf.constant(conf.lam),dtype=tf.float32)

	def _get_weights(self):
		return self.model.get_weights()

	def _get_posterior(self,sess):
		qmeans = []
		qstds = []
		for i,layers in enumerate(self.model.layers):
			try:
				q = layer.kernel_posterior
			except AttributeError:
				continue
			qmeans.append(q.mean())
			qstds.append(q.stddev())
		return sess.run([qmeans, qstds])

	def get_weights(self,sess):
		return self._get_weights()
		"""
		model_type = conf.model_type
		assert model_type in ['NN','CNN','BNN','CBNN','CBLN','CCBLN'], 'Model type is not valid'
		BNN = ['BNN','CBNN','CBLN','CCBLN']
		if model_type in BNN:
			return self._get_posterior(sess)
		else:
			return self._get_weights()
		"""

	def store_params(self,task_idx=None):
		if task_idx is not None:
			task_idx = 0

		net_params[task_idx] = self.get_weights()

	def set_weights(self,task_idx=None):
		if task_idx is None:
			task_idx = 0

		self.model.set_weights(net_params[task_idx])

	def get_model_info(self):
		#input_shape = conf.shape_of_sample
		output_shape = conf.num_classes
		num_layers = conf.num_layers
		hidden_dim = conf.hidden_dim

		TITLE = ('Model Info'.center(80,'#'))
		print(TITLE)
		print('Model Info: ')
		print('Input Shape (Holder, shape in configuration file): ', self.X_holder.shape)
		print('Output Shape: ', output_shape, self.y_holder.shape)
		print('Number of Hidden Layers: ', num_layers)
		print('Neurons per Hidden Layers: ', hidden_dim)
		print('Number of Tasks to be Trained: ', conf.num_tasks)
		for i in range(conf.num_tasks):
			print('	-- Task %d: '%i, conf.task_labels[i])
		print('#'*len(TITLE))



	def __call__(self):
		return self.model(self.X_holder)
