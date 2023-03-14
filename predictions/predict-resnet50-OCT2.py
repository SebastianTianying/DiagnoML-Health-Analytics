# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('/Users/seb/Downloads/medical-imaging-analytics/saved_model/resnet50-OCT-Binary-model')
# Check its architecture
import numpy as np
from keras.preprocessing import image

# predicting images
path = 'predict-pics/OCT-CNV.jpeg'

img = tf.keras.utils.load_img(path, target_size=(224, 224))
plt.imshow(img)
plt.show()

img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)
prediction = new_model.predict(img_preprocessed)
print(prediction)