
# coding: utf-8

# In[8]:
# TensorFlow wizardry
# import tensorflow as tf
# from keras import backend as k
# config = tf.ConfigProto()
#
# # Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True
#
# # Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 1
#
# # Create a session with the above options specified.
# k.tensorflow_backend.set_session(tf.Session(config=config))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# In[1]:

# =============================================================================
# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())
# =============================================================================
#input()


# In[2]:


# =============================================================================
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# =============================================================================
#input()

# In[3]:

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes




"""Test ImageNet pretrained DenseNet"""

import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K
from keras.models import Model
import os, os.path
import pandas as pd
import sys
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Activation
import tensorflow as tf


batch_size = 5







# =============================================================================
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# =============================================================================


# =============================================================================
# Model Formation
# =============================================================================

# We only test DenseNet-121 in this script for demo purpose
#sys.path.insert(0, '/home/raj/Desktop/Resnet152/notebooks')
#import resnet152_model
from resnet50 import ResNet50
weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

#featurefile = open('feature.txt', 'w+')
model = ResNet50()
model.load_weights(weights_path)




sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()



desiredOutputs = model.get_layer('flatten').output #or model.layers[n].outputs

#desiredOutputs = model.get_layer('avg_pool').output #or model.layers[n].outputs
# =============================================================================
# desiredOutputs = model.get_layer('relu5_blk').output #or model.layers[n].outputs
# =============================================================================

#desiredOutputs = MaxPooling2D((3,3), strides = 2, padding = 'valid')(desiredOutputs)
desiredOutputs = Dense(10, activation = 'softmax')(desiredOutputs)
partialModel = Model(model.inputs,desiredOutputs)
partialModel.load_weights('partialModelweights.h5')


for layer in partialModel.layers[:-20]:
    layer.trainable = False

for layer in partialModel.layers[-20:]:
    layer.trainable = True

# input()

#partialModel.load_weights('partialModelweights.h5')

print(get_model_memory_usage(batch_size,partialModel))
#input()

# =============================================================================
#
# =============================================================================






# In[4]:


###############################################################################################################
#image path and valid extensions
#imageDir = "/home/raj/Desktop/iPhone6splus" #specify your path here
#imageDir = "/home/raj/Desktop/Alina dd ko android camera bata khicheko photo har" #specify your path here
datasetdir = "C:\Final-Models" #specify your path here
testdatasetdir = "C:\Validation"

image_path_list = []
image_label_list = []
valid_image_extensions = [".jpg", ".jpeg"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

#create a list all files in directory and
#append files with a vaild extention to image_path_list

for phonemodel in os.listdir(os.path.join(datasetdir)):
    for file in os.listdir(os.path.join(datasetdir, phonemodel)):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue

        image_path_list.append(os.path.join(datasetdir,phonemodel, file))
        image_label_list.append(phonemodel)

# =============================================================================
# print(image_path_list)
# print(image_label_list)
# =============================================================================

print(len(image_path_list))
print(len(image_label_list))
#input()
#input()
###############################################################################################################

# =============================================================================
# Dataframe for features
# =============================================================================




# In[5]:


# =============================================================================
#
# # In[7]:
#
#
#
# from keras.utils.vis_utils import plot_model
# plot_model(partialModel, to_file='C:/Users/guita/Desktop/Mero/Resnet152/Models/Image Data Preparation/model.svg', show_shapes=True, show_layer_names=True)
# #plot_model(partialModel, to_file='/home/raj/Desktop/Resnet152/Models/MaxPool/Resnet152-Partial Model.svg', show_shapes=True, show_layer_names=True)
#
#
# # In[8]:
#
#
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# SVG(model_to_dot(partialModel, show_layer_names=True, show_shapes=True,rankdir='LR').create(prog='dot', format='svg' )) #TB - toptobottom LR-lefttoright
#
#
# # In[9]:
# =============================================================================

# =============================================================================
#
# from contextlib import redirect_stdout
# with open('/home/raj/Desktop/Resnet152/Models/MaxPool/FullModelSummary.txt','w') as fullModelsummaryfile:
#     with redirect_stdout(fullModelsummaryfile):
#         model.summary()
# with open('/home/raj/Desktop/Resnet152/Models/MaxPool/PartialModelSummary.txt','w') as partialModelsummaryfile:
#     with redirect_stdout(partialModelsummaryfile):
#         partialModel.summary()
#
# =============================================================================

# In[24]:


# =============================================================================
# #loop through image_path_list to open each image
# Images =np.array()
# for imagePath in image_path_list:
#
#     im = cv2.resize(cv2.imread(imagePath), (224, 224)).astype(np.float32)
#     #im = cv2.resize(cv2.imread('shark.jpg'), (224, 224)).astype(np.float32)
#
#     # Subtract mean pixel and multiple by scaling constant
#     # Reference: https://github.com/shicai/DenseNet-Caffe
#     im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
#     im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
#     im[:,:,2] = (im[:,:,2] - 123.68) * 0.017
#
#     if K.image_dim_ordering() == 'th':
#       # Transpose image dimensions (Theano uses the channels as the 1st dimension)
#       im = im.transpose((2,0,1))
#
#       # Use pre-trained weights for Theano backend
#       weights_path = '/home/raj/Desktop/Resnet152/notebooks/resnet152_weights_th.h5'
#     else:
#       # Use pre-trained weights for Tensorflow backend
#       weights_path = 'resnet152_weights_tf.h5'
#
#     # Insert a new dimension for the batch_size
#     im = np.expand_dims(im, axis=0)
#     Images.append(im)
#     print(len(Images))
#
# =============================================================================
    # Test pretrained model






# In[9]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

encoder = LabelEncoder()
encoder.fit(image_label_list)

Labeldataset = encoder.transform(image_label_list)
Labeldataset = np_utils.to_categorical(Labeldataset)


# In[10]:


from keras.preprocessing.image import ImageDataGenerator
ImageData = ImageDataGenerator().flow_from_directory(datasetdir, target_size=(224,224), classes= list(encoder.classes_), batch_size = batch_size)
TestData = ImageDataGenerator().flow_from_directory(testdatasetdir, target_size = (224,224), classes = list(encoder.classes_),batch_size = 8)

# =============================================================================
#
# # In[62]:
#
#
# value[:,:,:,0] = (value[:,:,:,0] - 103.94) * 0.017
# value[:,:,:,1] = (value[:,:,:,1] - 116.78) * 0.017
# value[:,:,:,2] = (value[:,:,:,2] - 123.68) * 0.017
#
#
# # In[17]:
#
# =============================================================================


# In[13]:


partialModel.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
#partialModel.summary()

#partialModel.predict(ImageData, verbose=1)

# =============================================================================
# 
# for phonemodel in os.listdir(os.path.join(datasetdir)):
#     for file in os.listdir(os.path.join(datasetdir, phonemodel)):
#         extension = os.path.splitext(file)[1]
#         if extension.lower() not in valid_image_extensions:
#             continue
#         im = cv2.resize(cv2.imread(os.path.join(datasetdir,phonemodel,file)), (224, 224)).astype(np.float32)
#         im = np.expand_dims(im, axis=0)
#         prediction = partialModel.predict(im, verbose =1)
#         print(phonemodel)
#         print(np.argmax(prediction))
# #         
# 
# =============================================================================



# In[14]:


result = partialModel.fit_generator(ImageData,steps_per_epoch = 10,validation_data=TestData, validation_steps = 10 , shuffle=True, verbose= True, epochs= 20)
partialModel.save('partialModelweights.h5')
# ================================= ============================================
#

# =============================================================================
# 
# testresult = partialModel.evaluate_generator(TestData)
# print(testresult)
# =============================================================================
