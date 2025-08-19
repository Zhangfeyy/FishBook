import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from twolayer import TwoLayerNet
import matplotlib.pylab as plt

(x_train, t_train), (x_test, t_test) = mnist.load_data()
#reshape
x_test = x_test.reshape(x_test.shape[0], -1)
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.astype('float32') / 255.0

#one-hot
t_train = to_categorical(t_train,10)
t_test = to_categorical(t_test,10)

train_loss_List = []
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	grad = network.numerical_gradient(x_batch, t_batch)
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate*grad[key]
	
	loss = network.loss(x_batch, t_batch)
	train_loss_List.append(loss)

x = np.arange(1, len(train_loss_List) + 1)
y = train_loss_List
plt.plot(x, y)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.show()





