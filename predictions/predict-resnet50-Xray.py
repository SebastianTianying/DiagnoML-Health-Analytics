import tensorflow as tf

new_model = tf.keras.models.load_model('saved_model/resnet50_model')
# Check its architecture
import numpy as np
from keras.preprocessing import image

# predicting images
path = '/Users/seb/Downloads/medical-imaging-analytics/predictions/predict-pics/NORMAL-1212407-0001.jpeg'
img = tf.keras.utils.load_img(path, target_size=(224, 224))
#plt.imshow(img)
#plt.show()

img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_batch = img_batch.astype('float32')
img = img_batch/255

prediction = new_model.predict(img)
print(prediction)

if prediction[0] <0.5:
    print("image shows PNEUMONIA")
else:
    print("image shows NORMAL")