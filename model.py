import os
import time
import shutil
import platform
from datetime import timedelta
import numpy as np
import tensorflow as tf 

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))

class 2StreamNet_VGG16(object):
	"""docstring for 2StreamNet"""
	def __init__(self, rgb_data, flow_data, num_classes, 
		sequence_length , weight_file, weights=None, sess=None):
		self.num_classes = num_classes
		self.rgb_data = rgb_data
		self.flow_data = flow_data
		self.sequence_length = sequence_length
		self.weight_file = weight_file

	# def conv3d(self,input,output_channels,kernel_size,strides=[1,1,1,1,1],padding='SAME'):
	# 	in_features = int(input.get_shape()[-1])
	# 	kernel = 

	def model(self):
		with tf.variable_scope('spatialNet'):
			with tf.variable_scope('conv1+pool1'):
				with tf.variable_scope('conv1_1_spatial'):
					weights = tf.get_variable("W",[3,3,3,64], 
						initializer=tf.contrib.layers.xavier_initialzier(),trainable=True)
					biases = tf.get_variable("b",[64],
						initializer=tf.constant_initializer(0.1),trainable=True)
					conv = tf.nn.conv2d(self.rgb_data,weights,strides=[1,1,1,1,1],padding='SAME')
					conv1_1_spatial = tf.nn.relu(conv+biases)

				with tf.variable_scope('conv1_2_spatial'):
					weights = tf.get_variable("W",[3,3,64,64], 
						initializer=tf.contrib.layers.xavier_initialzier(),trainable=True)
					biases = tf.get_variable("b",[64],
						initializer=tf.constant_initializer(0.1),trainable=True)
					conv = tf.nn.conv2d(conv1_1_spatial,weights,strides=[1,1,1,1,1],padding='SAME')
					conv1_2_spatial = tf.nn.relu(conv+biases)

				with tf.variable_scope('pool1_spatial'):
					pool1_spatial=tf.nn.max_pool(conv1_2_spatial,ksize=[1,2,2,1],
						strides=[1,2,2,1],padding='SAME')
			with tf.variable_scope('conv2+pool2'):
				with tf.variable_scope('conv2_2_spatial'):
					weights = tf.get_variable("W",[3,3,64,128], 
						initializer=tf.contrib.layers.xavier_initialzier(),trainable=True)
					biases = tf.get_variable("b",[128],
						initializer=tf.constant_initializer(0.1),trainable=True)
					conv = tf.nn.conv2d(pool1_spatial,weights,strides=[1,1,1,1,1],padding='SAME')
					conv2_1_spatial = tf.nn.relu(conv+biases)

				with tf.variable_scope('conv2_2_spatial'):
					weights = tf.get_variable("W",[3,3,128,128], 
						initializer=tf.contrib.layers.xavier_initialzier(),trainable=True)
					biases = tf.get_variable("b",[64],
						initializer=tf.constant_initializer(0.1),trainable=True)
					conv = tf.nn.conv2d(conv2_1_spatial,weights,strides=[1,1,1,1,1],padding='SAME')
					conv2_2_spatial = tf.nn.relu(conv+biases)

				with tf.variable_scope('pool2_spatial'):
					pool1_spatial=tf.nn.max_pool(conv2_2_spatial,ksize=[1,2,2,1],
						strides=[1,2,2,1],padding='SAME')
			with tf.variable_scope('conv3+pool3'):
				with tf.variable_scope('conv3_1_spatial'):
		            weights = tf.get_variable("W", [3,3,128,256], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [256], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(pool2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv3_1_spatial = tf.nn.relu(conv + biases)#[?,56,56,256]
		            
		        with tf.variable_scope('conv3_2_spatial'):
		            weights = tf.get_variable("W", [3,3,256,256],
		             initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [256], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv3_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv3_2_spatial = tf.nn.relu(conv + biases) #[?,56,56,256]
		            
		        with tf.variable_scope('conv3_3_spatial'):
		            weights = tf.get_variable("W", [3,3,256,256], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [256], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv3_2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv3_3_spatial = tf.nn.relu(conv + biases) #[?,56,56,256]
		            
		        with tf.variable_scope('pool3_spatial'):
		            pool3_spatial = tf.nn.max_pool(conv3_3_spatial, ksize=[1, 2, 2, 1], 
	            	strides=[1, 2, 2, 1], padding='SAME', name='pool3_spatial') #[?, 28,28,256]
			with tf.variable_scope('conv4+pool4'):
		        with tf.variable_scope('conv4_1_spatial'):
		            weights = tf.get_variable("W", [3,3,256,512], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(pool3_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv4_1_spatial = tf.nn.relu(conv + biases) #[?,28,28,512]
		            
		        with tf.variable_scope('conv4_2_spatial'):
		            weights = tf.get_variable("W", [3,3,512,512], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv4_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv4_2_spatial = tf.nn.relu(conv + biases) #[?,28,28,512]
		            
		        with tf.variable_scope('conv4_3_spatial'):
		            weights = tf.get_variable("W", [3,3,512,512], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv4_2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv4_3_spatial = tf.nn.relu(conv + biases) #[?,28,28,512]
		            
		        with tf.variable_scope('pool4_spatial'):
		            pool4_spatial = tf.nn.max_pool(conv4_3_spatial, ksize=[1, 2, 2, 1], 
		            strides=[1, 2, 2, 1], padding='SAME', name='pool4_spatial') #[?, 14,14,512]
		    with tf.variable_scope('conv5+pool5'):
		    	with tf.variable_scope('conv5_1_spatial'):
		            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(pool4_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv5_1_spatial = tf.nn.relu(conv + biases) #[?,14,14,512]
	            
		        with tf.variable_scope('conv5_2_spatial'):
		            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv5_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv5_2_spatial = tf.nn.relu(conv + biases) #[?,14,14,512]
		            
		        with tf.variable_scope('conv5_3_spatial'):
		            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv5_2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv5_3_spatial = tf.nn.relu(conv + biases) #[?,14,14,512]
	            
		        #with tf.variable_scope('pool5_spatial'):
		            #pool5_spatial = tf.layers.max_pooling2d(conv5_3_spatial, pool_size=[2,2], strides=(2,2), padding='valid') #[?, 7,7,512]

	    with tf.variable_scope('temporalNet'):
	    	with tf.variable_scope('conv1+pool1'):
	    		with tf.variable_scope('conv1_1_temporal'): # flow_imputs: [batch_size*nFramesPerVid, height, width, nStacks]  nStacks=20
		            weights = tf.get_variable("W", [3,3,20,64], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [64], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(self.flow_inputs, weights, 
		            	strides=[1, 1, 1, 1], padding='SAME')
		            conv1_1_temporal = tf.nn.relu(conv + biases)    #[?,224,224,64]
		        
		        with tf.variable_scope('conv1_2_temporal'):
		            weights = tf.get_variable("W", [3,3,64,64], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [64], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv1_1_temporal, 
		            	weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv1_2_temporal = tf.nn.relu(conv + biases) #[?,224,224,64]
		            
		        with tf.variable_scope('pool1_temporal'):
		            pool1_temporal = tf.nn.max_pool(conv1_2_temporal, ksize=[1, 2, 2, 1], 
		            strides=[1, 2, 2, 1], padding='SAME', name='pool1_temporal') #[?, 112,112,64]
			with tf.variable_scope('conv2+pool2'):
				with tf.variable_scope('conv2_1_temporal'):
		            weights = tf.get_variable("W", [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(pool1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv2_1_temporal = tf.nn.relu(conv + biases) #[?,112,112,128]
		            
		        with tf.variable_scope('conv2_2_temporal'):
		            weights = tf.get_variable("W", [3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv2_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv2_2_temporal = tf.nn.relu(conv + biases) #[?,112,112,128]
		            
		        with tf.variable_scope('pool2_temporal'):
		            pool2_temporal = tf.nn.max_pool(conv2_2_temporal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_temporal') #[?, 56,56,128]
			with tf.variable_scope('conv3+pool3'):
				with tf.variable_scope('conv3_1_temporal'):
		            weights = tf.get_variable("W", [3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(pool2_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv3_1_temporal = tf.nn.relu(conv + biases)#[?,56,56,256]
		            
		        with tf.variable_scope('conv3_2_temporal'):
		            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv3_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv3_2_temporal = tf.nn.relu(conv + biases) #[?,56,56,256]
		            
		        with tf.variable_scope('conv3_3_temporal'):
		            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv3_2_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv3_3_temporal = tf.nn.relu(conv + biases) #[?,56,56,256]
		            
		        with tf.variable_scope('pool3_temporal'):
		            pool3_temporal = tf.nn.max_pool(conv3_3_temporal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_temporal') #[?, 28,28,256]
			with tf.variable_scope('conv4+pool4'):
				with tf.variable_scope('conv4_1_temporal'):
		            weights = tf.get_variable("W", [3,3,256,512], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(pool3_temporal, weights, 
		            	strides=[1, 1, 1, 1], padding='SAME')
		            conv4_1_temporal = tf.nn.relu(conv + biases) #[?,28,28,512]
		            
		        with tf.variable_scope('conv4_2_temporal'):
		            weights = tf.get_variable("W", [3,3,512,512], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv4_1_temporal, weights, 
		            	strides=[1, 1, 1, 1], padding='SAME')
		            conv4_2_temporal = tf.nn.relu(conv + biases) #[?,28,28,512]
		            
		        with tf.variable_scope('conv4_3_temporal'):
		            weights = tf.get_variable("W", [3,3,512,512], 
		            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], 
		            	initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv4_2_temporal, weights, 
		            	strides=[1, 1, 1, 1], padding='SAME')
		            conv4_3_temporal = tf.nn.relu(conv + biases) #[?,28,28,512]
		            
		        with tf.variable_scope('pool4_temporal'):
		            pool4_temporal = tf.nn.max_pool(conv4_3_temporal, ksize=[1, 2, 2, 1], 
		            strides=[1, 2, 2, 1], padding='SAME', name='pool4_temporal') #[?, 14,14,512]
			with tf.variable_scope('conv5+pool5'):
				with tf.variable_scope('conv5_1_temporal'):
		            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(pool4_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv5_1_temporal = tf.nn.relu(conv + biases) #[?,14,14,512]
		            
		        with tf.variable_scope('conv5_2_temporal'):
		            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv5_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv5_2_temporal = tf.nn.relu(conv + biases) #[?,14,14,512]
		            
		        with tf.variable_scope('conv5_3_temporal'):
		            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
		            # Create variable named "biases".
		            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
		            conv = tf.nn.conv2d(conv5_2_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
		            conv5_3_temporal = tf.nn.relu(conv + biases) #[?,14,14,512]   此处往上?=batchsize*nFramesPerVid

		        #     conv5_3_reshape = tf.reshape(conv5_3_temporal,[-1,self.nFramesPerVid,14,14,512]) #[?,self.nFramesPerVid, 14, 14, 512]
		        # with tf.variable_scope('pool5_temporal'):
		        #     pool5_temporal = tf.layers.max_pooling3d(conv5_3_reshape, pool_size=[self.nFramesPerVid,2,2], strides=(2,2,2), padding='valid') #[?,1,7,7,512]

	    with tf.variable_scope('netFusion'):
	    	#np.concatenate((conv5_3_spatial, conv5_3_temporal), axis=3) # [?, 14, 14, 1024]
	    	spatial_temporal_concat = tf.concat([conv5_3_spatial, conv5_3_temporal],3) 
	        fusion_reshape = tf.reshape(spatial_temporal_concat,[-1, self.nFramesPerVid, 14, 14, 1024])   #[?, self.nFramesPerVid, 14, 14, 1024]
	        fusion_conv6 = tf.layers.conv3d(fusion_reshape, filters=512, kernel_size=[3,3,3], 
	        strides=(1,1,1), padding='same', activation=tf.nn.relu) #[?,self.nFramesPerVid,14,14,512]
	        pool3d = tf.layers.max_pooling3d(fusion_conv6, pool_size=[self.nFramesPerVid,2,2], 
	        strides=(2,2,2), padding='valid') # [?,1,7,7,512]  ?=batchsize 
	        pool3d_flat = tf.reshape(pool3d, [-1, 7*7*512])
	        with tf.variable_scope('fc6_spatial'):
	            fc6_W = tf.get_variable('W', [7*7*512, 4096], i
	            	nitializer=tf.contrib.layers.xavier_initializer(), trainable=True)
	            fc6_b = tf.get_variable('b',[4096], 
	            	initializer=tf.constant_initializer(0.1), trainable=True)
	            fusion_fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool3d_flat, fc6_W), fc6_b))
	        with tf.variable_scope('fc7_spatial'):
	            fc7_W = tf.get_variable('W', [4096, 4096], 
	            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
	            fc7_b = tf.get_variable('b',[4096], 
	            	initializer=tf.constant_initializer(0.1), trainable=True)
	            fusion_fc7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fusion_fc6, fc7_W), fc7_b))
	        with tf.variable_scope('fc8_spatial'):
	            fc8_W = tf.get_variable('W', [4096, self.nClasses], 
	            	initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
	            fc8_b = tf.get_variable('b',[self.nClasses], 
	            	initializer=tf.constant_initializer(0.1), trainable=True)
	            fusion_fc8 = tf.nn.bias_add(tf.matmul(fusion_fc7, fc8_W), fc8_b)

		return fusion_fc8

	def load_pretrained_model(self,session):
		weights_dict = np.load(self.weight_file,encoding='bytes')
		vgg_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 
		'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'] #, 'fc6', 'fc7']
        print("*********####################################################################******")
        for layer in vgg_layers:
            if layer != 'conv1_1':
                with tf.variable_scope(layer+"_spatial", reuse = True):
                    var = tf.get_variable('W', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_W']))
                    var = tf.get_variable('b', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_b']))
                with tf.variable_scope(layer+"_temporal", reuse = True):
                    var = tf.get_variable('W', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_W']))
                    var = tf.get_variable('b', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_b']))
            else:
                with tf.variable_scope(layer+"_spatial", reuse = True):
                    var = tf.get_variable('W', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_W']))
                    var = tf.get_variable('b', trainable = True)
                session.run(var.assign(weights_dict[layer+'_b']))
        vgg_fc_layers = ['fc6', 'fc7']
        for layer in vgg_fc_layers:
            with tf.variable_scope(layer+'_spatial', reuse = True):
                var = tf.get_variable('W', trainable=True)
                session.run(var.assign(weights_dict[layer+'_W']))
            with tf.variable_scope(layer+'_spatial', reuse = True):
                var = tf.get_variable('b', trainable=True)
                session.run(var.assign(weights_dict[layer+'_b']))

		