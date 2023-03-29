import tensorflow as tf
from tensorflow.keras import activations

_csc_fc_op_module = tf.load_op_library('/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/core/user_ops/csc_fc_op.so')

# def csc_fc(inputs, units, csc_c, csc_n, csc_f, csc_s, activation=None):
#     kernel_shape = (inputs.shape[-1], units)
#     bias_shape = (units,)

#     # kernel = tf.Variable(tf.random.normal(kernel_shape), trainable=True)
#     bias = tf.Variable(tf.zeros(bias_shape), trainable=True)

#     output = _csc_fc_op_module.csc_fc(inputs, kernel, bias, csc_c, csc_n, csc_f, csc_s)

#     if activation is not None:
#         return activations.get(activation)(output)
#     return output

def csc_fc(inputs, kernel, bias, csc_c, csc_n, csc_f, csc_s, activation=None):
    output = _csc_fc_op_module.csc_fc(inputs, kernel, bias, csc_c, csc_n, csc_f, csc_s)

    if activation is not None:
        return activations.get(activation)(output)
    return output

# print("Success!!!")