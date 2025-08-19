import pickle
from tensorflow.keras.datasets import mnist # the ds is in the cache
import numpy as np
from tensorflow.keras.utils import to_categorical


(x_train,t_train), (x_test, t_test) = mnist.load_data()
print(x_train.shape)
print(t_train.shape)

#reshape
x_test = x_test.reshape(x_test.shape[0], -1)
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.astype('float32') / 255.0

#one-hot
t_train_onehot = to_categorical(t_train,10)
t_test_onehot = to_categorical(t_test,10)

train_size = x_train.shape[0]
batch_size = 10
# select batch_size indice from the assigned numbers.
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# for one-hot
def cross_entropy_error(y,t):
	if y.ndim == 1:
		# avoiding vague shapes
		t = t.reshape(1, t.size) # 2-dim arrays, indicating 1 sample with size features
		y = y.reshape(1, y.size)
	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size
# for labels
def cross_entropy_error(y,t):
	if y.ndim == 1:
		# avoiding vague shapes
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	batch_size = y.shape[0] # col counts
	# np.arrange: to find the corresponding possibility to true labels
	# mimic the logic of one-hot
	return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size
