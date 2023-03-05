from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

#
from tensorflow.python.keras.engine.base_layer import Layer, InputSpec
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
import numpy as np
import tensorflow as tf

class CSC_FC(base.Layer):
	def __init__(self, num_outputs, activation):
		super(CSC_FC, self).__init__()
		self.num_outputs = num_outputs
		self.activation = activations.get(activation)

	def build(self, input_shape):
		self.kernal = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

	def call(self, inputs):
		outputs = tf.matmul(inputs, self.kernel)
		return self.activation(outputs)

# def cscFC(inputs, num_outputs, activation):
def cscFC(num_outputs, activation):
	layer = CSC_FC(num_outputs=num_outputs, activation=activation)

	# return layer(inputs)