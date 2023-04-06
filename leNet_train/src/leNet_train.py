import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow.python.framework import ops
import importlib
import sys
from tensorflow.lite.python.op_hint import OpHint
import datetime

csc_fc_lib_path = '/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/core/user_ops/csc_fc_op.so'

# Check if the custom op has already been imported
if 'csc_fc_module' not in sys.modules:
    print("Not found, importing CSC_FC module")
    csc_fc_module = tf.load_op_library(csc_fc_lib_path)
else:
    print("Found, reloading CSC_FC module")
    csc_fc_module = importlib.reload(sys.modules['csc_fc_module'])

csc_fc = csc_fc_module.csc_fc
csc_fc_grad = csc_fc_module.csc_fc_grad

@ops.RegisterGradient("CscFc")
def _csc_fc_grad_cc(op, grad):
    gradients = csc_fc_grad(op.inputs[0], op.inputs[1], grad, op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6])
    return gradients[0], gradients[1], None, None, None, None, None

class CSCFCLayer(tf.keras.layers.Layer):
    def __init__(self, csc_c, csc_n, csc_f, csc_s, **kwargs):
        super(CSCFCLayer, self).__init__(**kwargs)
        self.csc_c = csc_c
        self.csc_n = csc_n
        self.csc_f = csc_f
        self.csc_s = csc_s
        self.kernel = tf.Variable(tf.random.normal([csc_c, csc_n], stddev=0.05, dtype=tf.float32))
        self.bias = tf.Variable(tf.zeros([csc_n], dtype=tf.float32))

    def call(self, inputs):
        return (csc_fc(inputs, self.kernel, self.bias, self.csc_c, self.csc_n, self.csc_f, self.csc_s))

    def get_config(self):
        config = super(CSCFCLayer, self).get_config()
        config.update({
            'csc_c': self.csc_c,
            'csc_n': self.csc_n,
            'csc_f': self.csc_f,
            'csc_s': self.csc_s
        })
        return config

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

CSC_C = tf.Variable(0)
CSC_N = tf.Variable(0)
CSC_F = tf.Variable(0)
CSC_S = tf.Variable(0)

model = models.Sequential()
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
model.add(CSCFCLayer(csc_c=120, csc_n=84, csc_f=10, csc_s=1))
model.add(layers.Dense(10, activation='softmax'))
# model.add(CSCFCLayer(csc_c=84, csc_n=10, csc_f=42, csc_s=1))
model.summary()

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# log_dir = "../bin/updated/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "../bin/updated/logs/fit/leNet_with_CSC_12xComp_test3"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(x_train, y_train, batch_size=64, epochs=32, validation_data=(x_val, y_val), callbacks=[tensorboard_callback])
model.summary()
# model.save('../bin/models/')
tf.saved_model.save(model, export_dir="../bin/models/test_tflite/")

#generate acc graphs
fig, axs = plt.subplots(2, 1, figsize=(15,15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])
plt.savefig('../bin/misc/leNet_train_acc_test_2.png')

#tflite conversion
def representative_dataset():
    for data in x_test.take(100):
        yield [tf.dtypes.cast(data, tf.float32)]

saved_model_dir = "../bin/models/test_tflite/"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Provide a path to the TensorFlow Lite custom op library
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
    ("/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so", "CscFc")
]


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.experimental_new_converter = True

tflite_model = converter.convert()

# Save the TFLite model
print("-----------------------> Converter Success!!!")
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)