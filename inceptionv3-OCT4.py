import os
import tensorflow as tf
import keras

from keras import layers
from keras import models
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

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
        target_size=(150, 150),  # All images will be resized to 224x224 as required by alexnet
        batch_size=20)

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/Users/seb/Downloads/OCT/test',  # This is the source directory for training images
        target_size = (150, 150),  # All images will be resized to 224x224 as required by alexnet
        batch_size = 20,
        # Use binary labels
        shuffle=False)

print(train_generator.class_indices)

base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')#for layer in base_model.layers:

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(4, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
inc_history = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 20)

acc = inc_history.history['accuracy'][-1]
val_acc = inc_history.history['val_accuracy'][-1]
loss = inc_history.history['loss'][-1]
val_loss = inc_history.history['val_loss'][-1]

print("Training accuracy: ", acc)
print("Training loss: ", loss)

print("Validation accuracy: ", val_acc)
print("Validation loss: ", val_loss)

base_model.save('saved_model/inceptionv3-OCT-Binary-model')
