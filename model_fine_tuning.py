from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import scipy




def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 1), shuffle=True, batch_size=1, target_size=(32, 32),
              class_mode='binary'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='rgb',
                                   class_mode=class_mode, target_size=target_size)


BS = 32
TS = (32, 32)
train_batch = generator('Dataset2/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('Dataset2/test', shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(SPE, VS)

# img,labels= next(train_batch)
# print(img.shape)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# we dont need the FC layers
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))

#######FREEZED LAYERS#######
base_model.trainable = False

preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Defining the model
inputs = tf.keras.Input(shape=(32, 32, 3))
x = preprocess_input(inputs)
# false because of batch norm. layers
x = base_model(x, training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

history_freezed = model.fit(train_batch, epochs=2, validation_data=valid_batch)

#######FINE TUNING#######

# new model
base_model.trainable = True

# Fine-tune from this layer onwards
# fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
# layer.trainable = False


# Lower learning rate
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

history = model.fit(train_batch, validation_data=valid_batch, epochs=8, initial_epoch=history_freezed.epoch[-1] + 1)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-6), metrics=['accuracy'])

history = model.fit(train_batch, validation_data=valid_batch, epochs=15, initial_epoch=history.epoch[-1] + 1)

model.save('models/new_data3_inter_db.h5', overwrite=True)


