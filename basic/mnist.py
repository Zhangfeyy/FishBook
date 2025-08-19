from tensorflow.keras.datasets import mnist # the ds is in the cache
from PIL import Image
import numpy as np

(x_train, t_train), (x_test, t_test) = mnist.load_data()
print(x_train.shape) # 60000 pieces, size: 28px * 28px
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def img_show(img):
	pil_img = Image.fromarray(np.uint8(img)) # from array to pics and turn the pixels into 0-255(int)
	pil_img.save("temp_mnist.png")
	import subprocess
	subprocess.run(['start', 'temp_mnist.png'], shell= True) # start means open the file and call the shell to show the pic

img = x_train[0]
label = t_train[0]

print(label)

print(img.shape)
img = img.reshape(28,28)

img_show(img)





