import tensorflow as tf
import coremltools as ct 

model = ct.models.MLModel('saved_model/resnet152_coreML.mlpackage')
print("model", model)
import numpy as np

path = 'NORMAL-predict.jpeg'

from PIL import Image
sub = Image.open('NORMAL-predict.jpeg').resize((224, 224)).convert('RGB')

out_dict = model.predict({"resnet152_input": sub})
print("dict: ", out_dict)
print(out_dict['Identity'])

if out_dict["classlabel"] <0.5:
    print("image shows PNEUMONIA")
else:
    print("image shows NORMAL")