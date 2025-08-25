import numpy as np

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad

def cross_entropy_error(y,t):
	if y.ndim == 1:
		# avoiding vague shapes
		t = t.reshape(1, t.size) # 2-dim arrays, indicating 1 sample with size features
		y = y.reshape(1, y.size)
	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size

def softmax(x):
    # batch is a 2dim array
    if x.ndim == 2: 
        x = x.T # for broadcasting!
        x = x - np.max(x, axis=0) # along the axis0
        # or directly transpose np.max(x, axis=1).reshape(-1, 1)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T # transpose back
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
	return 1/(1 + np.exp(-x))

