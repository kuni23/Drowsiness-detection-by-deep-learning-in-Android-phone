from keras.models import load_model
import tensorflow as tf


model = load_model('models/new_data3_intra_db.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('lite_models/model_new_data3_intra_db.tflite', 'wb') as f:
  f.write(tflite_model)