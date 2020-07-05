import numpy as np
from utils.load_data import Load_data
import tensorflow as tf
from configuration import conf


class Baseline_loader(object):
	def __init__(self):
		try:
			load_func = getattr(load_data,'load_'+ conf.dataset_name)
			self.data = load_func(conf.is_conv)
		except:
			assert ValueError('Dataset is not available ... ')
		conf.num_samples = self.data['X_train'].shape[0]
		conf.shape_of_sample = self.data['X_train'].shape[1:]

	@property
	def num_classes(self):
		return len(np.unique(self.data['y_train']))

	@property
	def num_samples(self):
		return self.data['X_train'].shape[1]

	@property
	def shape_of_sample(self):
		return self.data['X_train'].shape[1:]

	@property
	def is_flatten(self):
		return len(self.data['X_train']) == 2
		

	def sample(self, dataset='train', batch_size=None):
		if batch_size is None:
			batch_size = conf.batch_size
		assert dataset in ['train', 'val', 'test']

		N = self.data['X_' + dataset].shape[0]
		idx_N = np.random.choice(N, batch_size, replace=False)

		images, labels = self.data['X_' + dataset][idx_N], self.data['y_' + dataset][idx_N]

		return images, labels

	def build_iterator(self, batch_size=None):
		if batch_size is None:
			batch_size = conf.batch_size
		self.init = {}
		with tf.name_scope('data'):
			train_data = tf.data.Dataset.from_tensor_slices(self.data['X_train']).batch(batch_size)
			test_data = tf.data.Dataset.from_tensor_slices(self.data['X_test']).batch(batch_size)
			iterator = tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
			img,label = iterator.get_next()
			self.init['train'] = iterator.make_initializer(train_data)
			self.init['test'] = iterator.make_initializer(test_data)

	def get_whole_dataset(self):
		return self.data

			

class Sequential_loader(object):
	def __init__(self):
		self.data = Load_data().load()
		self._task_idx = 0
		num_samples = 0
		for i in range(len(self.data)):
			num_samples += self.data[i]['X_train'].shape[0]
		conf.num_samples = num_samples

	@property
	def num_classes(self):
		num_classes = 0
		for i in range(len(self.data)):
			num_classes += len(np.unique(self.data[i]['y_train']))
		return num_classes

	@property
	def num_samples(self):
		return conf.num_samples

	@property
	def shape_of_sample(self):
		return self.data[0]['X_train'].shape[1:]

	@property
	def is_flatten(self):
		return len(self.data[0]['X_train']) == 2
		

	def sample(self, task_idx=None, dataset='train', batch_size=None,whole_set=False):
		if batch_size is None:
			batch_size = conf.batch_size
		if task_idx is None:
			task_idx = self._task_idx
		assert dataset in ['train', 'val', 'test']

		if whole_set:
			return self.data[task_idx]['X_' + dataset], self.data[task_idx]['y_' + dataset]

		N = self.data[task_idx]['X_' + dataset].shape[0]
		idx_N = np.random.choice(N, batch_size, replace=False)

		images, labels = self.data[task_idx]['X_' + dataset][idx_N], self.data[task_idx]['y_' + dataset][idx_N]
		if labels.ndim == 1:
			labels = labels[:,np.newaxis]
		return images, labels



	def _build_iterator(self, batch_size=None):
		if batch_size is None:
			batch_size = conf.batch_size
		self.data_init = {}
		with tf.name_scope('data'):
			train_data = tf.data.Dataset.from_tensor_slices((self.data[0]['X_train'],self.data[0]['y_train'])).batch(batch_size)
			test_data = tf.data.Dataset.from_tensor_slices((self.data[0]['X_test'],self.data[0]['y_test'])).batch(batch_size)
			iterator = tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
			self.img_holder,self.label_holder = iterator.get_next()
			self.data_init[0] = {}
			self.data_init[0]['train'] = iterator.make_initializer(train_data)
			self.data_init[0]['test'] = iterator.make_initializer(test_data)
			for i in range(1,len(self.data)):
				train_data = tf.data.Dataset.from_tensor_slices((self.data[0]['X_train'],self.data[0]['y_train'])).batch(batch_size)
				test_data = tf.data.Dataset.from_tensor_slices((self.data[0]['X_test'],self.data[0]['y_test'])).batch(batch_size)
				self.data_init[i] = {}
				self.data_init[i]['train'] = iterator.make_initializer(train_data)
				self.data_init[i]['test'] = iterator.make_initializer(test_data)

	@property
	def task_idx(self):
		return self._task_idx
	

	@task_idx.setter
	def task_idx(self,idx):
		self._task_idx = idx
		print('------------ Training Task Index : %d ------------'%self._task_idx)
	

	def initial_data(self,task_idx=None):
		if not hasattr(self,'data_init'):
			self._build_iterator()
		if task_idx is None:
			task_idx = self._task_idx

		return self.data_init[task_idx]

	def get_holder(self):
		if conf.enable_iterator:
			return self.img_holder, self.label_holder
		else:
			img_holder = tf.placeholder(dtype=tf.float32,shape=[None,self.shape_of_sample[0]])
			label_holder = tf.placeholder(dtype=tf.float32,shape=[None,conf.num_classes] if conf.enable_one_hot else [None,1])

			return img_holder, label_holder
	

	def get_whole_dataset(self):
		return self.data


