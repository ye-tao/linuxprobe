# code from http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html
# learning turn image to tfrecord
# read the cat vs dog dataset and convert it into train, val and test
# features include image and label
# one example is a pair of image and label
# the number of example is same as the number of images
from random import shuffle
import glob
import pdb 
import cv2 
import tensorflow as tf
import sys
import numpy as np

shuffle_data = True
cat_dog_train_path = 'dog_vs_cat/train/*jpg'

# read addresses and label from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'cat' in addr else 1 for addr in addrs] 
# 0 = Cat, 1 = Dog

# to shuffle data
# pdb.set_trace()
if shuffle_data:
	c = list(zip(addrs,labels))
	shuffle(c)
	addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]

val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

# create a TFRecords file
# A funtion to load images
def load_image(addr):
	# read an image and resize to (224,224)
	# cv2 load images as BGR, convert it to RGB
	img = cv2.imread(addr)
	img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.astype(np.float32)
	return img

# 2_Convert data to features
# 3_create a feature using tf.train.Feature and pass the converted data to it
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# write data into a TFRecords file
train_filename = 'train.tfrecords'

# 1_open a tfrecords file using tf.python_io.TFRecordWriter
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
	# print how many images are saved every 1000 images
	if not i%1000:
		print('Train data: {}/{}'.format(i, len(train_addrs)))
		sys.stdout.flush()

	# load the image
	img = load_image(train_addrs[i])

	label = train_labels[i]

	#Create a feature
	feature = {'train/label': _int64_feature(label),
				'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

	# 4_Create an example protocol buffer
	example = tf.train.Example(features = tf.train.Features(feature=feature))

	# 5_Serialize to string
	# 6_write on the file
	writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

# write data into a TFRecords file
val_filename = 'val.tfrecords'

# 1_open a tfrecords file using tf.python_io.TFRecordWriter
writer = tf.python_io.TFRecordWriter(val_filename)

for i in range(len(val_addrs)):
	# print how many images are saved every 1000 images
	if not i%1000:
		print('Val data: {}/{}'.format(i, len(val_addrs)))
		sys.stdout.flush()

	# load the image
	img = load_image(val_addrs[i])

	label = val_labels[i]

	#Create a feature
	feature = {'val/label': _int64_feature(label),
				'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

	# 4_Create an example protocol buffer
	example = tf.train.Example(features = tf.train.Features(feature=feature))

	# 5_Serialize to string
	# 6_write on the file
	writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

# write data into a TFRecords file
train_filename = 'test.tfrecords'

# 1_open a tfrecords file using tf.python_io.TFRecordWriter
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(test_addrs)):
	# print how many images are saved every 1000 images
	if not i%1000:
		print('Test data: {}/{}'.format(i, len(test_addrs)))
		sys.stdout.flush()

	# load the image
	img = load_image(test_addrs[i])

	label = test_labels[i]

	#Create a feature
	feature = {'test/label': _int64_feature(label),
				'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

	# 4_Create an example protocol buffer
	example = tf.train.Example(features = tf.train.Features(feature=feature))

	# 5_Serialize to string
	# 6_write on the file
	writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()