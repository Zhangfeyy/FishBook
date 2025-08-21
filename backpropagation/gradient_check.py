import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from b_twolayernet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = mnist.load_data()

#reshape
x_test = x_test.reshape(x_test.shape[0], -1)
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.astype('float32') / 255.0

#one-hot
t_train = to_categorical(t_train,10)
t_test = to_categorical(t_test,10)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
	# absolute difference
	diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
	print(key + ":" + str(diff))
