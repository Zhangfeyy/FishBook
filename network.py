import numpy as np
import matplotlib.pylab as plt
# # simple unit step function
# def step_function1(x):
# 	if x > 0:
# 		return 1
# 	else:
# 		return 0

# # array-accepted step function
# def step_function2(x):
# 	y = x > 0
# 	return y.astype(np.int64) # convert to int type array, here int must be specified as int64 or int32

# print("step function test")
# print(step_function2(np.array([1.0,4.0,-3.0])))

# # draw the step function

# def step_function3(x):
# 	return np.array(x>0, dtype = np.int64)

# x = np.arange(-5.0,5.0,0.1)
# y = step_function3(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1) # limit the scope of y-axis
# plt.show()

# # sigmoid function
# def sigmoid(x):
# 	return 1/(1 + np.exp(-x))
# print("sigmoid test")
# print(sigmoid(np.array([-1.0, 1.0, 2.0])))

# x = np.arange(-5.0,5.0,0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

# # ReLU
# def relu(x):
# 	return np.maximum(0,x)

# x = np.arange(-5.0,5.0,0.1)
# y = relu(x)
# plt.plot(x,y)
# plt.ylim(-1.0,5.0)
# plt.show()

# # multi-dimensional arrays (md-ary)
# B = np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(np.ndim(B)) #col
# print(B.shape) #row*col

# multiplication
A = np.array([[1,2,3],[4,5,6]]) #2*3
B = np.array([[1,2],[3,4],[5,6]]) #3*2
print(np.dot(A,B)) #2*2