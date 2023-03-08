import tensorflow as tf
import coremltools as ct 

model = tf.keras.models.load_model('saved_model/resnet152_model')
image_input = ct.ImageType(name="resnet152_input", shape=(1, 224, 224, 3,),)
class_labels = ['PNEUMONIA', 'NORMAL']
classifier_config = ct.ClassifierConfig(class_labels)

mlmodel = ct.convert(
    model, inputs=[image_input], classifier_config=classifier_config,
)

mlmodel.save('saved_model/resnet152_coreML.mlpackage')