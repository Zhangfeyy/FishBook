# list
# print("list\n")
# a = [1,2,3,4,5]
# print(a)
# print(len(a))
# print(a[0])
# print(a[0:2])
# print(a[1:])
# print(a[:3])
# print(a[:-1]) # till the last element(not included)
# print(a[:-1]) # till the last second element(not included)

# dictionary
# print("dictionary test")
# me = {'zhangfeyy': 'chinese'}
# print(me['zhangfeyy'])
# print(me)

# class
# print("class test")
# class person:
# 	def __init__(self,name):
# 		self.name = name
# 	def hello(self):
# 		print("sayhello" + self.name)
# 	def goodbye(self):
# 		print("goodbye" + self.name)
# me = person("zhangfeyy")
# me.hello()
# me.goodbye()

# # generate arrays in numpy
import numpy as np
# x = np.array([1.0,2.0,3.0])
# y = np.array([2.0, 4.0, 5.0])
# print(x)

# print(x + y)
# print(x / y)
# print(x * 2)

# # multiple dimension arrays
# A = np.array([[1,2], [3,4]])
# B = np.array([[-1,-2],[-3,-4]])
# print(A)
# print(A.shape)
# print(A.dtype)
# print(A * B)
# print(A + B)

# broadcast
# A = np.array([[1,2],[3,4]])
# B = np.array([[10,100], [10,100]])
# print(A * B)

# visit
x = np.array([[1,2], [3,4],[6,7]])
# print(x[0])
# print(x[0][1])

# # enumerate
# for row in x:
# 	print(row)

# # convert to 1-dimension array
# y = x.flatten()
# print(y)
# print(y[np.array([0,2,4])]) # generate a new ary in x
# print(y[y>3])

# draw the plot
import matplotlib.pyplot as plt

x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x,y1,label='sinx')
plt.plot(x,y2,label='cosx')
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()
