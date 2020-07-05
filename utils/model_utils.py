import tensorflow as tf
from configuration import conf
from keras.layers import Lambda


def mask_layer_by_task(task_input,input_tensor,name=None,return_mask=False):
	#print(task_input,[tf.shape(input_tensor)[0],1])
	#mask = K.tile(task_input,[1,input_tensor.shape[1]//5])
	mask = tf.expand_dims(task_input, axis=-1)
	mask = tf.tile(mask, multiples=[1, 1,input_tensor.shape[1]//conf.num_tasks]) if len(mask.shape) == 3 else tf.tile(mask, multiples=[1,input_tensor.shape[1]//conf.num_tasks])
	mask = tf.keras.layers.Flatten()(mask)
	if name is None:
		out = Lambda(lambda x: x*mask)(input_tensor)
	else:
		out = Lambda(lambda x: x * mask, name=name)(input_tensor)
	if return_mask:
		return out,mask
	else:
		return out


def mask_output_layer(task_input,input_tensor,name=None,output_dim=10, return_mask=False):
	#print(task_input,[tf.shape(input_tensor)[0],1])
	#mask = K.tile(task_input,[1,input_tensor.shape[1]//5])
	index = tf.argmax(task_input[0])
	new_mask = np.zeros(output_dim)
	for l in conf.task_labels[index]:
		new_mask[l] = 1.0
	mask = tf.tile(mask, multiples=[1, 1,input_tensor.shape[1]//conf.num_tasks]) if len(mask.shape) == 3 else tf.tile(mask, multiples=[1,input_tensor.shape[1]//conf.num_tasks])
	mask = tf.keras.layers.Flatten()(mask)
	if name is None:
		out = Lambda(lambda x: x*mask)(input_tensor)
	else:
		out = Lambda(lambda x: x * mask, name=name)(input_tensor)
	if return_mask:
		return out,mask
	else:
		return out