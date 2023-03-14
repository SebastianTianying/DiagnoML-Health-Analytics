# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('/Users/seb/Downloads/medical-imaging-analytics/saved_model/resnet50-OCT-Binary-model')
new_model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001), 
                            loss = 'binary_crossentropy', 
                            metrics = ['accuracy'])

# predicting images
#path = 'predict-pics/CNV-9499888-11.jpeg'
#path = 'predict-pics/OCT-NORMAL.jpeg' 
path = 'predict-pics/OCT-CNV-2.jpeg'

img = tf.keras.utils.load_img(path, target_size=(224, 224))
#img = tf.keras.utils.load_img(path)
#plt.imshow(img)
#plt.show()

img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_batch = img_batch.astype('float32')
img = img_batch/255
#img_preprocessed = preprocess_input(img_batch)
prediction = new_model.predict(img)
print(prediction[0][0])