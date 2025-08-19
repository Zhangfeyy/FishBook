import numpy as np

# one-hot expression
# the real value is the third element
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


# MSE
def mean_squared_error(y, t):
	return 0.5 * np.sum((y-t)**2)

# case1: ele3 has the biggest possiblity
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print("case1: " + str(mean_squared_error(np.array(y1), np.array(t))))

# case2: ele8 has the biggest possibility
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print("case2: " + str(mean_squared_error(np.array(y2), np.array(t))))


# Cross_entropy_error
def cross_entropy_error(y,t):
	delta = 1e-7
	return -np.sum(t * np.log(y + delta))
# delta is to control the case that when y = 0, log0 will be -infinitive
# print("case1: " + str(cross_entropy_error(np.array(y1), np.array(t))))
# print("case1: " + str(cross_entropy_error(np.array(y2), np.array(t))))

#mini-batch
