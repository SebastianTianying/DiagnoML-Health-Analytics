# TensorFlow and tf.keras
import tensorflow as tf
#Import Numpy, Keras Image and InceptionV3 libraries
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('/Users/seb/Downloads/medical-imaging-analytics/saved_model/inceptionv3-OCT-Binary-model')
# Check its architecture
import numpy as np
from keras.preprocessing import image

# predicting images
path = 'predict-pics/OCT-CNV-2.jpeg'

img = tf.keras.utils.load_img(path, target_size=(150, 150))
#plt.imshow(img)
#plt.show()

img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = inception_v3.preprocess_input(img_batch)
prediction = new_model.predict(img_preprocessed)
print(prediction.shape)
print(prediction[0][0])
