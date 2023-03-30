import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Lambda
from tensorflow.python.framework import ops
import importlib
import sys
from tensorflow.lite.python.op_hint import OpHint

# Define the path to the custom op library
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

# @ops.RegisterGradient("CscFc")
# def _csc_fc_grad_cc(op, grad):
#     return csc_fc_grad(op.inputs[0], op.inputs[1], grad, op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6])
@ops.RegisterGradient("CscFc")
def _csc_fc_grad_cc(op, grad):
    gradients = csc_fc_grad(op.inputs[0], op.inputs[1], grad, op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6])
    return gradients[0], gradients[1], None, None, None, None, None
    # return gradients[0], gradients[1], gradients[2], gradients[3], gradients[4], None, None


# Define the custom op wrapper function
def csc_fc_wrapper(x, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s):
    return csc_fc(x, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s)

def csc_fc_tflite(x, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s):
    return csc_fc(x, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s)

class CscFcOpHint(OpHint):
    def __init__(self, layer, **kwargs):
        super(CscFcOpHint, self).__init__(function_name="CscFc", **kwargs)
        self.layer = layer

    def __call__(self, inputs, outputs):
        input_shape = tf.constant(inputs.shape.as_list()[1:], dtype=tf.int32)
        output_shape = tf.constant(outputs.shape.as_list()[1:], dtype=tf.int32)
        self.add_hint("input_shape", input_shape)
        self.add_hint("output_shape", output_shape)
        return outputs

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
        # csc_fc_output = csc_fc(inputs, self.kernel, self.bias, self.csc_c, self.csc_n, self.csc_f, self.csc_s)
        return (csc_fc(inputs, self.kernel, self.bias, self.csc_c, self.csc_n, self.csc_f, self.csc_s))
        # return CscFcOpHint(self)(inputs, csc_fc_output)

    def get_config(self):
        config = super(CSCFCLayer, self).get_config()
        config.update({
            'csc_c': self.csc_c,
            'csc_n': self.csc_n,
            'csc_f': self.csc_f,
            'csc_s': self.csc_s
        })
        return config

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the model with the custom CSC_FC layer
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(28 * 28,)),
#     tf.keras.layers.Lambda(csc_fc_wrapper, arguments={'kernel_tensor': kernel_tensor, 'bias_tensor': bias_tensor, 'csc_c': 784, 'csc_n': 128, 'csc_f': 8, 'csc_s': 1}),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28 * 28,)),
    CSCFCLayer(csc_c=784, csc_n=128, csc_f=8, csc_s=1),
    tf.keras.layers.Dense(10, activation='softmax')
])


# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=32, batch_size=32, validation_data=(x_test, y_test))

# Save the model with signatures
# input_signature = tf.TensorSpec(shape=[None, 28*28], dtype=tf.float32)
# signatures = {
#     'serving_default': model.get_concrete_function(input_signature)
# }
# tf.saved_model.save(model, export_dir="../bin/models/test_tflite/", signatures=signatures)
tf.saved_model.save(model, export_dir="../bin/models/test_tflite/")

# Convert the modified model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("../bin/models/test_tflite/")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

# Find the OpHint for the custom operation in the graph
hints = CscFcOpHint.get_all_hinted_operations(converter.get_concrete_function().graph)
csc_fc_hint = next((hint for hint in hints if hint.op_type == "CscFc"), None)

# Add the custom OpHint to the converter
if csc_fc_hint:
    converter.target_ops = [csc_fc_hint]

tflite_model = converter.convert()

# Save the TFLite model
with open("../bin/models/test_tflite/model.tflite", "wb") as f:
    f.write(tflite_model)