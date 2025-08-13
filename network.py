import numpy as np
import matplotlib.pylab as plt
# simple unit step function
def step_function1(x):
	if x > 0:
		return 1
	else:
		return 0

# array-accepted step function
def step_function2(x):
	y = x > 0
	return y.astype(np.int64) # convert to int type array, here int must be specified as int64 or int32

print("step function test")
print(step_function2(np.array([1.0,4.0,-3.0])))

# draw the step function

def step_function3(x):
	return np.array(x>0, dtype = np.int64)

x = np.arange(-5.0,5.0,0.1)
y = step_function3(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1) # limit the scope of y-axis
plt.show()

# sigmoid function
def sigmoid(x):
	return 1/(1 + np.exp(-x))
print("sigmoid test")
print(sigmoid(np.array([-1.0, 1.0, 2.0])))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

# ReLU
def relu(x):
	return np.maximum(0,x)

x = np.arange(-5.0,5.0,0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-1.0,5.0)
plt.show()

# multi-dimensional arrays (md-ary)
B = np.array([[1,2],[3,4],[5,6]])
print(B)
print(np.ndim(B)) #col
print(B.shape) #row*col

 # inner product
A = np.array([[1,2,3],[4,5,6]]) #2*3
B = np.array([[1,2],[3,4],[5,6]]) #3*2
print(np.dot(A,B)) #2*2

# matrix inner product in network
X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])
Y = np.dot(X,W)
print(Y)

# A = XW + B
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

A1 = np.dot(X,W1) + B1
Z1 = sigmoid(A1)
print("Layer1")
print(A1)
print(Z1)

W2 = np.array([[0.1,0.4], [0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
print("Layer2")
print(A2)
print(Z2)

def identity_function(x):
	return X
W3 = np.array([[0.1,0.3], [0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3) + B3
Z3 = identity_function(A3)
print("Layer3")
print(A3)
print(Z3)

# network function

def init_network():
	network = {} # define a dictionary
	network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
	network['b1'] = np.array([0.1,0.2,0.3])
	network['w2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
	network['b2'] = np.array([0.1,0.2])
	network['w3'] = np.array([[0.1,0.3],[0.2,0.4]])
	network['b3'] = np.array([0.1,0.2])
	return network;

# forward: processing direction
def forward(network,x):
	w1,w2,w3 = network['w1'],network['w2'], network['w3']
	b1,b2,b3 = network['b1'],network['b2'],network['b3']

	a1 = np.dot(x,w1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1,w2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2,w3) + b3
	y = a3
	return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)

# softmax - original
def softmax(a):	
	exp_a = np.exp(a)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y

a = np.array([0.3,2.9,4.0])
print("softmax")
print(softmax(a))

# softmax -revised
def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y

a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
print(np.sum(y))