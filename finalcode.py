# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:11:48 2018

@author: guita
"""



# =============================================================================
# Import necessary library files
# =============================================================================

import cv2
import numpy as np
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.models import Model
import os, os.path
import pandas as pd
import sys
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf


# =============================================================================
# Avoid Truncated Error
# =============================================================================

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



# =============================================================================
# Set Directories
# =============================================================================
model_weight_path='partialModelweights.h5'
train_dataset_dir = "/media/rose/Windows/Final-Models"
test_dataset_dir = "/media/rose/Windows/Validation"


# =============================================================================
# Set variables
# =============================================================================

Trainable_layers_Number = 2
TrainBatchSize = 4
TestBatchSize = 4
sgd = SGD(lr=1e-2, decay=1e-6,momentum=0.9, nesterov=True)
adam = Adam(lr=0.001)


# =============================================================================
# Import Resnet Model and create newModel and loadweights
# =============================================================================
from resnet50 import ResNet50
model = ResNet50()
# model.load_weights("/media/rose/Windows/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
desiredOutputs = model.get_layer('flatten').output
# desiredOutputs = Dropout(0.2)(desiredOutputs)
# desiredOutputs = Dense(1000)(desiredOutputs)
# desiredOutputs = Activation('relu')(desiredOutputs)
desiredOutputs = Dense(10)(desiredOutputs)
desiredOutputs = Activation('softmax')(desiredOutputs)
partialModel = Model(model.inputs,desiredOutputs)
#print(partialModel.weights)
#partialModel.load_weights("weights201805261000.h5")

partialModel.summary()

#partialModel.load_weights(model_weight_path)

# =============================================================================
# Configure layers as trainable
# =============================================================================

# for layer in partialModel.layers[:-Trainable_layers_Number]:
#     layer.trainable = False
#
# for layer in partialModel.layers[-Trainable_layers_Number:]:
#     layer.trainable = True

for layer in partialModel.layers:
    layer.trainable = True
    print(layer.name,layer.trainable)

# =============================================================================
# Compile the Model
# =============================================================================
partialModel.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])




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
print(get_model_memory_usage(TrainBatchSize,partialModel))

# =============================================================================
# List of Images directory and the list of label of the Images
# =============================================================================
# =============================================================================
# Create List for the label of the phonemodel
# =============================================================================

image_label_list = []

for phonemodel in os.listdir(train_dataset_dir):
   image_label_list.append(phonemodel)

print(len(image_label_list))

# =============================================================================
#
# =============================================================================
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(image_label_list)
# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================




# =============================================================================
# ImageGenerator for train and test
# =============================================================================
from keras.preprocessing.image import ImageDataGenerator
TrainImageDataGenerator = ImageDataGenerator(validation_split=0.3).flow_from_directory(train_dataset_dir, target_size=(224,224), classes=list(encoder.classes_), batch_size=TrainBatchSize)
TestImageDataGenerator = ImageDataGenerator().flow_from_directory(test_dataset_dir, target_size=(224,224), classes=list(encoder.classes_), batch_size=TestBatchSize)



# =============================================================================
# Fit the Model and save the weights
# =============================================================================
TrainingResult = partialModel.fit_generator(TrainImageDataGenerator, steps_per_epoch=len(TrainImageDataGenerator), validation_data=TestImageDataGenerator, validation_steps=len(TestImageDataGenerator), shuffle=True, verbose=1, epochs=7)

partialModel.save_weights('fulltrainweightsfile.h5')


# =============================================================================
# Plot the trainresult
# =============================================================================
import matplotlib.pyplot as plt
plt.plot(TrainingResult.history['acc'])
plt.plot(TrainingResult.history['val_acc'])
plt.title("Model Accuracy")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(TrainingResult.history['loss'])
plt.plot(TrainingResult.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend(['train','test'], loc='upper left')
plt.show()


TestTrainData = partialModel.evaluate_generator(TrainImageDataGenerator, verbose = 1)
TestTestData = partialModel.evaluate_generator(TestImageDataGenerator, verbose = 1)

print("Result on Train Data:"+ str(TestTrainData))
print("Result on Test Data:" + str(TestTestData))
