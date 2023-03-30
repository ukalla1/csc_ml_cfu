import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Lambda

# from csc_fc_op import csc_fc


class CSCFCLayer(tf.keras.layers.Layer):
    def __init__(self, units, csc_c, csc_n, csc_f, csc_s, activation=None, **kwargs):
        super(CSCFCLayer, self).__init__(**kwargs)
        self.units = units
        self.csc_c = csc_c
        self.csc_n = csc_n
        self.csc_f = csc_f
        self.csc_s = csc_s
        self.activation = tf.keras.activations.get(activation)
        from csc_fc_op import csc_fc  # Move the import statement here
        self.csc_fc = csc_fc

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", (input_shape[-1], self.units), initializer="glorot_uniform")
        self.bias = self.add_weight("bias", (self.units,), initializer="zeros")

    # def call(self, inputs):
    #     csc_fc_output = self.csc_fc(inputs, self.kernel, self.bias, self.csc_c, self.csc_n, self.csc_f, self.csc_s)
    #     if self.activation is not None:
    #         return self.activation(csc_fc_output)
    #     return csc_fc_output
    def call(self, inputs):
        csc_fc_output = self.csc_fc(inputs, self.kernel, self.bias, self.csc_c, self.csc_n, self.csc_f, self.csc_s)
        if self.activation is not None:
            return self.activation(csc_fc_output)
        return csc_fc_output


# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the model with the custom CSC_FC layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28 * 28,)),
    CSCFCLayer(units=128, csc_c=784, csc_n=128, csc_f=776, csc_s=1, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Save the modified model
model.save("../bin/models/test_tflite/")

# Convert the modified model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("../bin/models/test_tflite/")
tflite_model = converter.convert()

# Save the TFLite model
with open("../bin/models/test_tflite/model.tflite", "wb") as f:
    f.write(tflite_model)
