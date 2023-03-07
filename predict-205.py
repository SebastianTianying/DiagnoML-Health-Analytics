import tensorflow as tf

new_model = tf.keras.models.load_model('saved_model/resnet152_model')

# Check its architecture
new_model.summary()

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# predicting images
path = 'BACTERIA-predict.jpeg'
img = tf.keras.utils.load_img(path, target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
plt.imshow(x/255.)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = new_model.predict(images, batch_size=10)
print(classes[0])

if classes[0]<0.5:
    print("image shows PNEUMONIA")
else:
    print("image shows NORMAL")