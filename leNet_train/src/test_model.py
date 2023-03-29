# import tensorflow as tf
# from tensorflow.python.framework import ops
# import numpy as np

# # Disable v2 behavior for compatibility
# tf.compat.v1.disable_v2_behavior()

# # Load the shared library
# csc_fc_module = tf.load_op_library('/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/core/user_ops/csc_fc_op.so')

# @ops.RegisterGradient("CscFc")
# def _csc_fc_grad(op, grad):
#     return csc_fc_module.csc_fc_grad(op.inputs[0], op.inputs[1], grad,
#                                       op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6])

# @tf.function
# def csc_fc_(input_tensor, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s):
#     return csc_fc_module.csc_fc(input_tensor, kernel_tensor, bias_tensor,
#                                  csc_c, csc_n, csc_f, csc_s)

# # Create sample input tensors
# input_np = np.random.rand(16, 32).astype(np.float32)
# kernel_np = np.random.rand(32, 64).astype(np.float32)
# bias_np = np.random.rand(64).astype(np.float32)

# # Convert numpy arrays to TensorFlow tensors
# input_tensor = tf.convert_to_tensor(input_np)
# kernel_tensor = tf.convert_to_tensor(kernel_np)
# bias_tensor = tf.convert_to_tensor(bias_np)

# # Set custom op parameters
# csc_c = 1
# csc_n = 32
# csc_f = 3
# csc_s = 1

# # Call the custom op
# output = csc_fc_(input_tensor, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s)

# # Evaluate the result
# print(output.numpy())

########################################################################################################

# import tensorflow as tf
# from tensorflow.python.framework import ops
# import numpy as np
# import importlib
# import sys

# # Define the path to the custom op library
# csc_fc_lib_path = '/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/core/user_ops/csc_fc_op.so'

# # Check if the custom op has already been imported
# if 'csc_fc_module' not in sys.modules:
#     print("Not found, importing CSC_FC module")
#     csc_fc_module = tf.load_op_library(csc_fc_lib_path)
# else:
#     print("Found, reloading CSC_FC module")
#     csc_fc_module = importlib.reload(sys.modules['csc_fc_module'])

# csc_fc = csc_fc_module.csc_fc
# csc_fc_grad = csc_fc_module.csc_fc_grad

# @ops.RegisterGradient("CscFc")
# def _csc_fc_grad_cc(op, grad):
#     return csc_fc_grad(op.inputs[0], op.inputs[1], grad, op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6])

# # Create sample input tensors
# input_np = np.random.rand(16, 32).astype(np.float32)
# kernel_np = np.random.rand(32, 64).astype(np.float32)
# bias_np = np.random.rand(64).astype(np.float32)

# # Convert numpy arrays to TensorFlow tensors
# input_tensor = tf.constant(input_np)
# kernel_tensor = tf.constant(kernel_np)
# bias_tensor = tf.constant(bias_np)

# # Set custom op parameters
# csc_c = 1
# csc_n = 32
# csc_f = 3
# csc_s = 1

# # Call the custom op
# output = csc_fc(input_tensor, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s)

# # Example of using the custom op in a model training
# # Create a simple model
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(32,)),
#     tf.keras.layers.Lambda(csc_fc, arguments={'kernel_tensor': kernel_tensor, 'bias_tensor': bias_tensor, 'csc_c': csc_c, 'csc_n': csc_n, 'csc_f': csc_f, 'csc_s': csc_s}),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Create random data for training
# x_train = np.random.rand(1000, 32).astype(np.float32)
# y_train = np.random.randint(0, 10, 1000).astype(np.int32)

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=16)

##################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import importlib
import sys

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

@ops.RegisterGradient("CscFc")
def _csc_fc_grad_cc(op, grad):
    return csc_fc_grad(op.inputs[0], op.inputs[1], grad, op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6])

# Create sample input tensors
input_np = np.random.rand(16, 32).astype(np.float32)
kernel_np = np.random.rand(32, 64).astype(np.float32)
bias_np = np.random.rand(64).astype(np.float32)

# Convert numpy arrays to TensorFlow tensors
input_tensor = tf.constant(input_np)
kernel_tensor = tf.constant(kernel_np)
bias_tensor = tf.constant(bias_np)

# Set custom op parameters
csc_c = 1
csc_n = 32
csc_f = 3
csc_s = 1

# Call the custom op
output = csc_fc(input_tensor, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s)

# Define the custom op wrapper function
def csc_fc_wrapper(x, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s):
    return csc_fc(x, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(32,)),
    tf.keras.layers.Lambda(csc_fc_wrapper, arguments={'kernel_tensor': kernel_tensor, 'bias_tensor': bias_tensor, 'csc_c': csc_c, 'csc_n': csc_n, 'csc_f': csc_f, 'csc_s': csc_s}),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create random data for training
x_train = np.random.rand(1000, 32).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.int32)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=16)
