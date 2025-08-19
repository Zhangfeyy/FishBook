import numpy as np

def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y

def numerical_gradient(f,x):
	h = 1e-4
	# generate the array with same shape
	grad = np.zeros_like(x)

	# flatten first!!!
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			tmp_val = x[i,j]
			x[i, j] = tmp_val + h
			fxh1 = f(x)

			x[i, j] = tmp_val -h
			fxh2 = f(x)

			grad[i, j] = (fxh1-fxh2) / (2*h)
			x[i, j] = tmp_val

	return grad

def cross_entropy_error_one(y,t):
	if y.ndim == 1:
		# avoiding vague shapes
		t = t.reshape(1, t.size) # 2-dim arrays, indicating 1 sample with size features
		y = y.reshape(1, y.size)
	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size

class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2,3)
		# initializing by Gaussian Distribution
	def predict(self,x):
		return np.dot(x, self.W)
	def loss(self,x,t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error_one(y,t)

		return loss


net  = simpleNet()
print(net.W)
x = np.array([0.6,0.9])
p = net.predict(x)
print("maximum index:" + str(np.argmax(p)))
t = np.array([0,0,1])
print("loss:" + str(net.loss(x,t)))

# virtual function
def f(W):
	return net.loss(x,t)
# f = lambda w:net.loss(x,t)

dW = numerical_gradient(f,net.W)
print("w:" + str(dW))


