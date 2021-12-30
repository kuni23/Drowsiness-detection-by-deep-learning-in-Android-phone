from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import scipy



def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=1, target_size=(32, 32),
              class_mode='binary'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='rgb',
                                   class_mode=class_mode, target_size=target_size)


BS = 32
TS = (32, 32)
train_batch = generator('Dataset3/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('Dataset3/test', shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(SPE, VS)

# img,labels= next(train_batch)
# print(img.shape)


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
outputs = tf.keras.layers.Flatten()(x)

model = tf.keras.Model(inputs, outputs)

model.summary()


def extract_features(loader, batch_number):
    features, labels = [], []
    i = 0

    for batch, targets in loader:

        batch_features = model(batch)

        features.append(batch_features.numpy())
        labels.append(targets)

        i = i + 1

        if i == batch_number:
            break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


# Extracting features
train_features, train_labels = extract_features(train_batch, SPE)
valid_features, valid_labels = extract_features(valid_batch, VS)

print(f'train features are {train_features.shape}')
print(f'valid features are {valid_features.shape}')

# Classification, logistic regression
classifier = LogisticRegression(C=1, solver='liblinear')

# Fit the logistic regression classifier on the training features
classifier.fit(train_features, train_labels)

# make predictions using the logistic regression classifier on the validation features
y_hat = classifier.predict(valid_features)

accuracy = metrics.accuracy_score(valid_labels, y_hat)
print(f'Accuracy: {accuracy:0.3f}')
