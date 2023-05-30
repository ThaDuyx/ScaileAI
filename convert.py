import coremltools as ct
import tensorflow as tf

tf_model_path = 'weights-improvement-198-3.1080-bigger.hdf5'
tf_model = tf.keras.models.load_model(tf_model_path)

mlmodel = ct.convert(tf_model)

mlmodel.save('weights-improvement-198-3.1080-bigger.mlmodel')