#read the tfrecords file created by write_tfrecord.py

import tensorflow as tf
import numpy as np 
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import pdb

data_path = 'train.tfrecords' #address to save the tfrecords file

with tf.Session() as sess:
	feature = {'train/image': tf.FixedLenFeature([], tf.string),
				'train/label': tf.FixedLenFeature([], tf.int64)}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path],num_epochs=1)

	#Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example,features=feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features["train/image"], tf.float32)

	# Cast label data into int32
	label = tf.cast(features['train/label'],tf.int32)

	# Reshape image data into the original shape
	image = tf.reshape(image,[224,224,3]) # a single image

	# Any preprocessing here ...

	# Creates batches by randomly suffling tensors
	images, labels = tf.train.shuffle_batch([image, label], batch_size=10,capacity=30,
		num_threads=1, min_after_dequeue=10)
	# image, label denote a format for features,shuffle_batch gathered the mini-batch 

	# Initialize all global and local variables
	init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(init_op)

	# Create a coordinator and run all QueueRunner objects
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	for batch_index in range(5):
		img, lbl = sess.run([images, labels])
		# img= sess.run(image)

		img = img.astype(np.uint8)
		# pdb.set_trace()
		# for j in range(6):
		# 	plt.subplot(2,3,j+1)
		# 	plt.imshow(img[j,...])
		# 	plt.title('cat' if lbl[j]==0 else 'dog')

		# plt.show()

	# Stop the threads
	coord.request_stop()

	# Wait for threads to Stop
	coord.join(threads)
	sess.close()