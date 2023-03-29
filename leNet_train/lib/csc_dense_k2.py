# import tensorflow.compat.v2 as tf
import tensorflow as tf

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.dtensor import utils
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from tensorflow.python.layers import base

import numpy as np

# isort: off
from tensorflow.python.util.tf_export import keras_export


# @keras_export("keras.layers.CSC_FC")
@tf.keras.utils.register_keras_serializable()
class CSC_FC(tf.keras.layers.Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). These are all attributes of
    `Dense`.
    Note: If the input to the layer has a rank greater than 2, then `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors).  The output in this case will have
    shape `(batch_size, d0, units)`.
    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    When a popular kwarg `input_shape` is passed, then keras will create
    an input layer to insert before the current layer. This can be treated
    equivalent to explicitly defining an `InputLayer`.
    Example:
    >>> # Create a `Sequential` model and add a Dense layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(tf.keras.layers.Dense(32))
    >>> model.output_shape
    (None, 32)
    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    @utils.allow_initializer_layout
    def __init__(
        self,
        units,
        # my_filter,                                #replacing this with the actual params
        CSC_C,
        CSC_N,
        CSC_F,
        CSC_S,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(CSC_FC, self).__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        # self.my_filter = my_filter                #replacing this with the actual params
        self.CSC_C = CSC_C
        self.CSC_N = CSC_N
        self.CSC_F = CSC_F
        self.CSC_S = CSC_S
        

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

        ############################change self.kernel here############################
        
        self.CustomMatrix = np.full((self.CSC_C, self.CSC_N), 0)
        for f in range (0, self.CSC_F):
            self.CustomMatrix [0, f*self.CSC_S]= 1
        for c in range (1, self.CSC_C):
            for n in range (0, self.CSC_N):
                self.CustomMatrix [c, n] = self.CustomMatrix [c-1, (n-1)%self.CSC_N]
        self.CustomMatrix = tf.convert_to_tensor(self.CustomMatrix, dtype=tf.float32)

        # assert self.kernel.shape == self.my_filter.shape, "Filter mulitplied with a different shape array "+str(self.my_filter.shape) +" "+str(self.kernel.shape)
        assert self.kernel.shape == self.CustomMatrix.shape, "Filter mulitplied with a different shape array "+str(self.CustomMatrix.shape) +" "+str(self.kernel.shape)

        # ## Kernel of a layer is a tf.Variable. To change its value, use the assign method.
        # ##mod to handle the custom_matrix generated internally
        # self.kernel.assign(tf.multiply(self.kernel, CustomMatrix))

        ###############################################################################

    # @tf.function(input_signature=[tf.TensorSpec([None, None, None, 1])])
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # print(f"test_inputs: {inputs}")

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension
            # (last dimension not ragged), we can flatten the input and restore
            # the ragged dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError(
                    "Dense layer only supports RaggedTensors when the "
                    "innermost dimension is non-ragged. Received: "
                    f"inputs.shape={inputs.shape}."
                )
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1]
                    )

        rank = inputs.shape.rank
        # ############################change self.kernel here############################
        
        # CustomMatrix = np.full((self.CSC_C, self.CSC_N), 0)
        # for f in range (0, self.CSC_F):
        #     CustomMatrix [0, f*self.CSC_S]= 1
        # for c in range (1, self.CSC_C):
        #     for n in range (0, self.CSC_N):
        #         CustomMatrix [c, n] = CustomMatrix [c-1, (n-1)%self.CSC_N]
        # CustomMatrix = tf.convert_to_tensor(CustomMatrix, dtype=tf.float32)

        # # assert self.kernel.shape == self.my_filter.shape, "Filter mulitplied with a different shape array "+str(self.my_filter.shape) +" "+str(self.kernel.shape)
        # assert self.kernel.shape == CustomMatrix.shape, "Filter mulitplied with a different shape array "+str(CustomMatrix.shape) +" "+str(self.kernel.shape)
        
        # ## Kernel of a layer is a tf.Variable. To change its value, use the assign method.
        # # self.kernel.assign(tf.multiply(self.kernel, self.my_filter))

        # ## Kernel of a layer is a tf.Variable. To change its value, use the assign method.
        # ##mod to handle the custom_matrix generated internally
        self.kernel.assign(tf.multiply(self.kernel, self.CustomMatrix))

        # ###############################################################################
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul
            # operation for large sparse input tensors. The op will result in a
            # sparse gradient, as opposed to
            # sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id
                # per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding
                # lookup as a matrix multiply. We split our input matrix into
                # separate ids and weights tensors. The values of the ids tensor
                # should be the column indices of our input matrix and the
                # values of the weights tensor can continue to the actual matrix
                # weights.  The column arrangement of ids and weights will be
                # summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed
                # explanation of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    self.kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = tf.matmul(a=inputs, b=self.kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the input shape of a Dense layer "
                "should be defined. Found None. "
                f"Received: input_shape={input_shape}"
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super(CSC_FC, self).get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(
                    self.bias_constraint
                ),
                "CSC_C": self.CSC_C,
                "CSC_N": self.CSC_N,
                "CSC_F": self.CSC_F,
                "CSC_S": self.CSC_S,
                # "CSC_C": constraints.serialize(
                #     self.CSC_C
                # ),
                # "CSC_N": constraints.serialize(
                #     self.CSC_N
                # ),
                # "CSC_F": constraints.serialize(
                #     self.CSC_F
                # ),
                # "CSC_S": constraints.serialize(
                #     self.CSC_S
                # ),
            }
        )
        return config

# class CSC_FC(Dense, Layer):
#     def __init__(
#         self,
#         units,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         **kwargs
#     ):
#         super(CSC_FC, self).__init__(
#             units=units,
#             activation=None,
#             use_bias=True,
#             kernel_initializer="glorot_uniform",
#             bias_initializer="zeros",
#             kernel_regularizer=None,
#             bias_regularizer=None,
#             activity_regularizer=None,
#             kernel_constraint=None,
#             bias_constraint=None,
#             **kwargs)

## csc layer: trying to replicate mortezas efforts
# def cscFC(units, my_filter, activation):
#     layer = CSC_FC(units=units, my_filter=my_filter, activation=activation, use_bias=False)
#     return layer


#csc layer: movign the csc params into the lib, to bake the csc params into the tflite model
def cscFC(units, activation, CSC_C, CSC_N, CSC_F, CSC_S):
    '''
        units,
        # my_filter,                                #replacing this with the actual params
        C,
        N,
        F,
        S,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    '''
    layer = CSC_FC(units=units, activation=activation, use_bias=False, CSC_C=CSC_C, CSC_N=CSC_N, CSC_F=CSC_F, CSC_S=CSC_S)
    return layer