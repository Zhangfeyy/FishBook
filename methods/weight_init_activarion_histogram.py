import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def ReLu(x):
	return np.maximum(0, x)

def tanh(x):
	return np.tanh(x)

# 1000 data with 100 features, drawn from std normal distribution
input_data = np.random.randn(1000,100)

node_num = 100 # hidden neutron size
hidden_layer_size = 5
activation = {}

x = input_data

for i in range(hidden_layer_size):
	if i != 0:
		x = activation[i - 1]
	
	w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)

	a = np.dot(x,w)
	z = ReLu(a)
	activation[i] = z

for i, a in activation.items():
	# 1 row, len column, current subplot position
	plt.subplot(1,len(activation),i+1)
	plt.title(str(i+1) + "-layer")
	# hide y-axis
	if i != 0:plt.yticks([],[])
	# flatten to see all the values, 30 bins, show output within (0,1)
	plt.hist(a.flatten(),30,range=(0,1))

plt.show()


