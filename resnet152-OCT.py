import os
import pickle
import tensorflow as tf
import keras

from keras import layers
from keras import models
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

import numpy as np

np.random.seed(1000)

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255, rotation_range = 40, width_shift_range = 0.2,
                                   height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2,
                                   horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/Users/seb/Downloads/OCT/train',  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 224x224 as required by alexnet
        batch_size=20)

print(train_generator.class_indices)

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/Users/seb/Downloads/OCT/test',  # This is the source directory for training images
        target_size = (224, 224),  # All images will be resized to 224x224 as required by alexnet
        batch_size = 20,
        shuffle=False)

base_model = ResNet152(input_shape=(224, 224,3), include_top=False, weights="imagenet")
#for layer in base_model.layers:
#    layer.trainable = False

base_model = Sequential()
base_model.add(ResNet152(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(4, activation='softmax'))

base_model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
resnet_history = base_model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 50)

acc = resnet_history.history['accuracy'][-1]
val_acc = resnet_history.history['val_accuracy'][-1]
loss = resnet_history.history['loss'][-1]
val_loss = resnet_history.history['val_loss'][-1]

print("Training accuracy: ", acc)
print("Training loss: ", loss)

print("Validation accuracy: ", val_acc)
print("Validation loss: ", val_loss)

base_model.save('saved_model/resnet152-OCT-model')
