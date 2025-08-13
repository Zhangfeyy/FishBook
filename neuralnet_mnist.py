import pickle
from tensorflow.keras.datasets import mnist # the ds is in the cache
import numpy as np

# sigmoid function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y

def get_data():
	(x_train, t_train), (x_test, t_test) = mnist.load_data()
	# flattening is necessary since the network can only handle the 1d array!
	# shape[0] is the number of pictures, -1 means automatically calculating the second dim
	# so we keep the first dim, and calculate: 10000*28*28/10000 = 784
	x_test = x_test.reshape(x_test.shape[0], -1)
	# normalizing, to narrow the scope and stablize the data
	x_test = x_test.astype('float32') / 255.0
	return x_test, t_test

def init_network():
	with open("sample_weight.pkl", 'rb') as f:
		network = pickle.load(f)
		# pickle can save the object in the running progress
		# here the f saved the weights of the network
	return network

def predict(network, x):
	w1,w2,w3 = network['W1'], network['W2'], network['W3']
	b1,b2,b3 = network['b1'], network['b2'], network['b3']
	a1 = np.dot(x, w1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, w2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, w3) + b3
	y = softmax(a3)

	return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y) # index of the element with the biggest possiblity
	if p == t[i]:
		accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# # a small trick in the book
# x, _= get_data()
# # _ is to ignore the return value we dont need

#batch processing
batch_size = 100


for i in range(0, len(x), batch_size):
	x_batch = x[i:i+batch_size]
	y_batch = predict(network, x_batch)
	p = np.argmax(y_batch, axis=1)
	accuracy_cnt += np.sum(p == t[i:i+batch_size])
# i in range(start, end, step)
# arr[i:i+step]: abstract the elements and generate a new list
# axis: search the max value along the axis (in matrix, axis0 is the row direction, axis1 is the col direction)

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))




