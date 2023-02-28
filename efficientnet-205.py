import efficientnet.keras as efn
import os

import numpy as np
import tensorflow as tf
from keras.layers import Dropout

from keras.layers import Dense, Flatten
from tensorflow import keras
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator

np.random.seed(1000)

train_positive_dir = os.path.join('train/PNEUMONIA')

train_negative_dir = os.path.join('train/NORMAL')

valid_positive_dir = os.path.join('test/PNEUMONIA')

valid_negative_dir = os.path.join('test/NORMAL')

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255, rotation_range = 40, width_shift_range = 0.2,
                                   height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2,
                                   horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'train',  # This is the source directory for training images
        classes = ['PNEUMONIA', 'NORMAL'],
        target_size=(224, 224),  # All images will be resized to 224x224 as required by alexnet
        batch_size=20,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'test',  # This is the source directory for training images
        classes = ['PNEUMONIA', 'NORMAL'],
        target_size = (224, 224),  # All images will be resized to 224x224 as required by alexnet
        batch_size = 20,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

base_model = efn.EfficientNetB0(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
predictions = Dense(1, activation="sigmoid")(x)
model_final = tf.keras.models.Model(base_model.input, predictions)

model_final.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),loss='binary_crossentropy',metrics=['accuracy'])

eff_history = model_final.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 20)

acc = eff_history.history['accuracy'][-1]
val_acc = eff_history.history['val_accuracy'][-1]
loss = eff_history.history['loss'][-1]
val_loss = eff_history.history['val_loss'][-1]

print("Training accuracy: ", acc)
print("Training loss: ", loss)

print("Validation accuracy: ", val_acc)
print("Validation loss: ", val_loss)