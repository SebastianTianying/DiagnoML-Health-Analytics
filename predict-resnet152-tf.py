import tensorflow as tf

new_model = tf.keras.models.load_model('saved_model/resnet152_model')
# Check its architecture
import numpy as np
from keras.preprocessing import image

# predicting images
path = 'VIRUS-predict-2.jpeg'
img = tf.keras.utils.load_img(path, target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = new_model.predict(images, batch_size=10)
print("classes: ", classes)

if classes[0] <0.5:
    print("image shows PNEUMONIA")
else:
    print("image shows NORMAL")