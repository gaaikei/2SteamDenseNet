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

	def conv3d(self,input,output_channels,kernel_size,strides=[1,1,1,1,1],padding='SAME'):
		in_features = int(input.get_shape()[-1])
		kernel = 

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
	    	with tf.variable_scope('conv2+pool2'):
	    	with tf.variable_scope('conv3+pool3'):



		