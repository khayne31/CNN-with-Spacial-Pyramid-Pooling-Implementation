
import theano as T
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.tensor.signal.pool import max_pool_2d_same_size
from PIL import Image
import numpy as np
import time
import sys

start = time.time()
for i in range(1):
	img = Image.open("images/Dogs/dog"+str(18)+".jpg")
	data = list(img.getdata())
	red_values = [x[0] for x in data]
	blue_values = [x[2] for x in data]
	green_values = [x[1	] for x in data]
	red_pixels = np.asarray(red_values, dtype = "int32").reshape((img.size[1], img.size[0]))
	blue_pixels = np.asarray(blue_values, dtype = "int32").reshape((img.size[1],img.size[0]))
	green_pixels = np.asarray(green_values, dtype = "int32").reshape((img.size[1], img.size[0]))
	#img_array = np.asarray(img, dtype = "int32")


	tensor1 = T.shared(np.matrix("0 0 0; 0 1 0; 0 0 0"))
	tensor2 = T.shared(np.matrix("0 0 0; 0 -1 0; 0 0 0"))
	tensor = T.shared(np.matrix("0 0 0; 0 5 0; 0 0 0"))
	tensor3 = T.shared(np.matrix("1 0 -1; 0 0 0; -1 0 1"))
	t1 = np.asarray((tensor.eval(), tensor1.eval(), tensor2.eval(), tensor3.eval()))
	#print(t1.shape)
	stride = (1,1)
	conv1 = conv2d(input = red_pixels, filters = t1, subsample = stride)
	conv2 = conv2d(input = green_pixels, filters = t1, subsample = stride)
	conv3 = conv2d(input = blue_pixels, filters = t1, subsample = stride)

	red_feature_map_layer1 = conv1.eval()
	blue_feature_map_layer1  = conv2.eval()
	green_feature_map_layer1  = conv3.eval()
	red_feature_map_layer1  = conv1.eval().reshape(red_feature_map_layer1 .shape[0], 
												   red_feature_map_layer1 .shape[2],
												   red_feature_map_layer1.shape[1])#feature map 1 from layer 1
	blue_feature_map_layer1 = conv2.eval().reshape(blue_feature_map_layer1.shape[0],
												   blue_feature_map_layer1.shape[2], 
												   blue_feature_map_layer1.shape[1])#feature map 2 from layer 1
	green_feature_map_layer1 = conv3.eval().reshape(green_feature_map_layer1.shape[0],
												    green_feature_map_layer1.shape[2], 
												    green_feature_map_layer1.shape[1])#feature map 3 from layer 1
	#shape((r,g,b), # filters, x, y)
	fmap_collection = np.asarray((red_feature_map_layer1, blue_feature_map_layer1, green_feature_map_layer1), dtype = "uint8")

	temp_list2 = []
	for j in range(fmap_collection.shape[1]):
		temp_list1 = []
		for i in range(red_feature_map_layer1.shape[1]):
			temp_list1.append(list(zip(red_feature_map_layer1[j][i], blue_feature_map_layer1[j][i], green_feature_map_layer1[j][i])))
	#array which holds the resulting feauture maps for each the r,g and b layers of the image after going through convolution(of shape(x,y,(r,g,b))) 
		temp = np.asarray(temp_list1, dtype = "uint8").reshape(red_feature_map_layer1.shape[2], red_feature_map_layer1.shape[1], 3)
		temp_list2.append(temp)
	#shape(#filters, x, y, (r,g,b))
	fmap_collection = np.asarray((temp_list2), dtype = "uint8").reshape(red_feature_map_layer1.shape[0],
																   red_feature_map_layer1.shape[2], 
																   red_feature_map_layer1.shape[1], 3)

	#chnage indices to get diffrent values for the filter from 0-#filters
	new_img = Image.fromarray(fmap_collection[0])
	new_img.show()
	sys.stdout.flush()

	#		RELU
	#relu now holds the feture maps corresponding with each filter of shape(# filters, width, height, (r,g,b))
	relu = T.tensor.nnet.relu(fmap_collection)
	#change the index to get diffrent value for the filter from 0-#filters
	relu = np.asarray(relu[0], dtype = "uint8").reshape(fmap_collection[0].shape)#tensor with the each feature map from layer 1 going through RELU
	relu_img = Image.fromarray(relu)
	relu_img.show()

	# 		MAX POOLING
	max_pool1 = max_pool_2d_same_size(relu.reshape(relu.shape[2], relu.shape[0], relu.shape[1]), patch_size = (3,3))
	poolresult = max_pool1.eval();
	
	#an array holding the feature maps after they go through max pooling of shape(x,y,(r,g,b))
	new_result = np.asarray(poolresult, dtype = "uint8").reshape(poolresult.shape[1], poolresult.shape[2], poolresult.shape[0])
	max_pool_img = Image.fromarray(new_result)
	max_pool_img.show()
	
	#		CONV LAYER 2
	data_layer2 = list(max_pool_img.getdata())
	red_values_layer2 = [x[0] for x in data_layer2]
	blue_values_layer2 = [x[2] for x in data_layer2]
	green_values_layer2 = [x[1] for x in data_layer2]
	red2 = np.asarray(red_values_layer2, dtype = "int32").reshape((max_pool_img.size[1], max_pool_img.size[0]))
	blue2 = np.asarray(blue_values_layer2, dtype = "int32").reshape((max_pool_img.size[1], max_pool_img.size[0]))
	green2 = np.asarray(green_values_layer2, dtype = "int32").reshape((max_pool_img.size[1], max_pool_img.size[0]))

	conv_layer2_1 = conv2d(red2.astype("int32"), filters = tensor, subsample = stride)
	conv_layer2_2 = conv2d(blue2.astype("int32"), filters = tensor, subsample = stride)
	conv_layer2_3 = conv2d(green2.astype("int32"), filters = tensor, subsample = stride)

	red_result_2 = conv_layer2_1.eval()
	blue_result_2 = conv_layer2_3.eval()
	green_result_2 = conv_layer2_2.eval()
	
	result_layer2 = np.asarray((red_result_2, green_result_2, blue_result_2), dtype = "uint8")

	new_layer2 = []
	for i in range(red_result_2.shape[0]):
		new_layer2.append(list(zip(red_result_2[i], green_result_2[i], blue_result_2[i])))
	result_layer2 = np.asarray(new_layer2, dtype = "uint8").reshape(red_result_2.shape[0], red_result_2.shape[1], 3)

	conv2_img = Image.fromarray(result_layer2)
	#conv2_img.show()

	#		RELU 2
	relu2 = T.tensor.nnet.relu(result_layer2)
	relu2 = np.asarray(relu2, dtype = "uint8").reshape(result_layer2.shape)
	act2 = Image.fromarray(relu2)
	#act2.show()
	
	#		2ND POOLING LAYER
	#print(relu2.shape)
	max_pool2 = max_pool_2d_same_size(relu2.reshape(3, relu2.shape[0], relu2.shape[1]), patch_size = (3,3))
	#max_pool2 = pool_2d(relu2.reshape(3, relu2.shape[0], relu2.shape[1]) ,ws = (2,2), ignore_border = True, mode = "max" )
	poolresult2 = max_pool2.eval();#holds each of the feature maps after two rounds of conv, relu and pooling

	new_result_layer2 = np.asarray(poolresult2, dtype = "uint8").reshape(poolresult2.shape[1], poolresult2.shape[2], poolresult2.shape[0])
	#print(poolresult2[0])
	max_pool_img2 = Image.fromarray(new_result_layer2)
	max_pool_img2.show() 

	spp_input_maps = (poolresult2[0], poolresult2[1], poolresult2[2])



print(str(time.time()-start))

# -TODO look more in to whether or not the max pooling a) makes sense for what the output is b) if it doesnt try and find a way to make it make sense.
# -Look into whether or not the out puts from conv, relu, and pooling shpuld be stored sepreatly as three feture maps(one for each rgb value) and whether or not it 
#  will make a diffrence when it comes to training. 
# -Look into spactal pooling layer once this is done
# -Look into the use of variables for the stride length, filter size, filter values, patch_size, 
# -Once we have variables and the spacial pooling layer look into connecting it into to a fully connect neural network
# -Do bacpropogation