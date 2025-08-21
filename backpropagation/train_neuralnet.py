import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from b_twolayernet import TwoLayerNet

(x_train,t_train), (x_test, t_test) = mnist.load_data()

# reshape
x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
x_test = x_test.astype('float32') / 255.0

#one-hot
t_train = to_categorical(t_train, 0)
t_test = to_categorical(t_test, 0)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	grad = network.gradient(x_batch,t_batch)
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]
	
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print(train_acc, test_acc)
