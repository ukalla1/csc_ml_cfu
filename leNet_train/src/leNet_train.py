import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np

from csc_dense_k2 import cscFC

global ct

def getCustomMatrix(C, N, F, S):
  CustomMatrix = np.full((C, N), 0)
  for f in range (0, F):
    CustomMatrix [0, f*S]= 1
  for c in range (1, C):
    for n in range (0, N):
      CustomMatrix [c, n] = CustomMatrix [c-1, (n-1)%N]

  # CustomMatrix = tf.convert_to_tensor(CustomMatrix, dtype=tf.float32)
  return CustomMatrix

#load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#print(x_test.shape)

x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]])/255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]])/255
# print(f"trs: {x_train.shape}, tts: {x_test.shape}")

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
# print(f"trs: {x_train.shape}, tts: {x_test.shape}")

x_val = x_train[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = x_train[:-2000,:,:,:]
y_train = y_train[:-2000]

# model = models.Sequential()
model = tf.keras.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='tanh'))
model.add(layers.Flatten())
# model.add(layers.Dense(84, activation='tanh'))
# model.add(layers.Dense(10, activation='softmax'))
model.add(cscFC(units=84, my_filter=getCustomMatrix(120, 84, 16, 2), activation='tanh'))
model.add(cscFC(units=10, my_filter=getCustomMatrix(84, 10, 8, 1), activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))
model.summary()
model.save('../bin/models/leNet_trained_test.h5')

fig, axs = plt.subplots(2, 1, figsize=(15,15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])
plt.savefig('../bin/misc/leNet_train_acc_test.png')