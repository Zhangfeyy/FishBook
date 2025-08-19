import numpy as np
import matplotlib.pylab as plt

# derivatives
def numerical_diff(f,x):
	h = 10e-50
	return (f(x+h) - f(x)) /h

def central_diff(f,x):
	h = 1e-4
	return (f(x+h) - f(x -h)) / (2 * h)

def function_1(x):
	return 0.01*x**2 + 0.1 *x

x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
print(central_diff(function_1,5))
print(central_diff(function_1,10))

# # partial derivatives
def function_2(x):
	return np.sum(x**2)
def function_tmp1(x):
	return x*x + 0
print(central_diff(function_tmp1,3))

def numerical_gradient(f,x):
	h = 1e-4
	# generate the array with same shape
	grad = np.zeros_like(x)

	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = tmp_val + h
		fxh1 = f(x)

		x[idx] = tmp_val -h
		fxh2 = f(x)

		grad[idx] = (fxh1-fxh2) / (2*h)
		x[idx] = tmp_val

	return grad
	
print(numerical_gradient(function_2, np.array([3.0,4.0])))

def gradient_descent(f, init_x, lr=0.01, step_num=100):
	x = init_x
	for idx in range(step_num):
		grad = numerical_gradient(f,x)
		x -= lr*grad
	return x

# step number: repeating times

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2, init_x = init_x, lr=0.1, step_num=100))
print(function_2(gradient_descent(function_2, init_x = init_x, lr=0.1, step_num=100)))