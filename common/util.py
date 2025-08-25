import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
	N,C,H,W = input_data.shape
	# / gives a decimal result but // gives an int result
	out_h = (H + 2*pad -filter_h) //stride + 1
	out_w = (W + 2*pad -filter_w) //stride + 1

	img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant')
	# sequently add 0 to N, C (no paddings) and 1 to H, W
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
	# Creates an empty array to hold the patches that the filter will slide over.
	# out_h, out_w: how many times the filter can slide vertically and horizontally

	# the loop is iterating in the filter not the sliding
	for y in range(filter_h):
		y_max = y+stride*out_h # calculate the ending positions
		for x in range(filter_w):
			x_max = x + stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
	
	col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
	return col