from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras import Sequential,layers
import scipy





def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=1, target_size=(24, 24),
              class_mode='binary'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale',
                                   class_mode=class_mode, target_size=target_size)


BS = 32
TS = (24, 24)
train_batch = generator('Dataset2/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('Dataset2/test', shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(SPE, VS)

# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    layers.Conv2D(24, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(48, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(),
    layers.Dense(48, activation='relu'),
    layers.Dense(1, activation='sigmoid')

])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_batch, validation_data=valid_batch, epochs=20, steps_per_epoch=SPE, validation_steps=VS)

model.save('models/new_data2_inter_db.h5', overwrite=True)


#print("First model evaluation!")
#model = load_model('models/new_data.h5')
#model.evaluate(valid_batch, steps=VS)

#print("Second model evaluation!")
#model = load_model('models/cnnCat2.h5')
#model.evaluate(valid_batch, steps=VS)

print("Second model evaluation!")
model = load_model('models/new_data2_inter_db.h5')
model.evaluate(valid_batch, steps=VS)



