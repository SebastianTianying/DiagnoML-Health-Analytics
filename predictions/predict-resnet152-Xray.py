# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('saved_model/resnet152_model')
# Check its architecture
import numpy as np
from keras.preprocessing import image

# predicting images
path = 'predict-pics/BACTERIA-predict.jpeg'

img = tf.keras.utils.load_img(path, target_size=(224, 224))
plt.imshow(img)
plt.show()

img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_batch = img_batch.astype('float32')
img = img_batch/255

prediction = new_model.predict(img)
print(prediction)