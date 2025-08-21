import numpy as np
from ..common.functions import * 

# switch in circus
class Relu:
	def __init__(self):
		self.mask = None
	
	# create a Boolean array to save the positions of x <= 0.
	# [True, True, True, False, False]
	def forward(self,x):
		self.mask = (x <= 0)
		out = x.copy()
	# set the marked positions values as 0
		out[self.mask] = 0

		return out
	
	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout

		return dx

class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = 1/ (1 + np.exp(-x))
		self.out = out
		return out
	
	def backward(self, dout):
		dx = dout * (1.0 -self.out) * self.out

		return dx

if __name__ == "__main__":
	X_dot_W = np.array([[0,0,0],[10,10,10]])
	B = np.array([1,2,3])

	Y  = X_dot_W + B
	print(Y) # B's broadcasting to each row
	'''
	 array([[ 1,  2,  3],
       		[11, 12, 13]])
	'''

	dY = np.array([[1,2,3],[4,5,6]])
	dB = np.sum(dY, axis=0)

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = B
		self.x = None
		self.dW = None
		self.db = None
	
	def forward(self,x):
		self.x = x
		out = np.dot(x, self.W) + self.b

		return out
	
	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis = 0)

		return dx

class SoftmaxWithLoss:
	def __inint__(self):
		self.loss - None
		self.y = None
		self.t = None

	def forward(self,x,t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y,self.t)

		return self.loss
	
	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size
		# to average the error to each single data
