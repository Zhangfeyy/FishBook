import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import *

# batch-size: 10, channel: 1, height:28, width: 28
x = np.random.rand(10,1,28,28)
# access the first item(data)
x[0].shape # (1, 28, 28)
# the second item
x[1].shape # (1, 28, 28)

# access data in the first channel of the first item
x[0][0].shape(28,28)
# x[0,0]

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1, 5,5,stride = 1, pad = 0)
print(col1.shape) # (9,75) -> 3*5*5
x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2,5,5,stride=1, pad=0)
print(col2.shape) # (90,75)


