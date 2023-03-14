# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('/Users/seb/Downloads/medical-imaging-analytics/saved_model/resnet50-OCT-model')
# Check its architecture
import numpy as np
from keras.preprocessing import image

# predicting images
#path = 'predict-pics/OCT-CNV-2.jpeg'
#path = 'predict-pics/OCT-NORMAL.jpeg' 
#path = 'predict-pics/DRUSEN-228939-46.jpeg' 
path = 'predict-pics/DME-15307-8.jpeg'

img = tf.keras.utils.load_img(path, target_size=(224, 224))
#plt.imshow(img)
#plt.show()

img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_batch = img_batch.astype('float32')
img = img_batch/255

prediction = new_model.predict(img)
print(prediction)

max_val = max(prediction[0])
index = np.where(prediction[0] == max_val)
print(index[0][0])
if index[0][0] == 3:
    print("normal")
elif index[0][0] == 0:
    print("CNV detected")
elif index[0][0] == 1:
    print("DME detected")
elif index[0][0] == 2:
    print("DRUSEN detected")
