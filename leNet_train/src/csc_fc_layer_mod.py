import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import InputSpec
from tensorflow.python.ops import custom_gradient
import numpy as np

# Register custom gradient
@tf.RegisterGradient("CSC_FC")
def _csc_fc_tf_grad(op, grad):
    input, weights, biases, csc_c, csc_n, csc_f, csc_s = op.inputs
    grad_input = tf.matmul(grad, weights, transpose_b=True)
    grad_weights = tf.matmul(input, grad, transpose_a=True)
    if biases is not None:
        grad_biases = tf.reduce_sum(grad, axis=0)
    else:
        grad_biases = None
    return grad_input, grad_weights, grad_biases, None, None, None, None

# @tf.RegisterGradient("CscFc")
# def _csc_fc_grad(op, grad):
#     # Implement the gradient for your custom operation here
#     return grad

def csc_fc_kernel(i_, k_, c_, n_, f_, s_):
    print(f"in custom MatMul...")
    with tf.GradientTape() as tape:
        tape.watch(i_)
        tape.watch(k_)
        input_shape = tf.shape(i_)
        batch_size = input_shape[0]
        input_size = input_shape[1]
        output_size = k_.shape[1]

        print(f"Printing the inputs and the output shapes:\n")
        print(f"input_shape: {input_size}")
        print(f"output shape: {output_size}")

        # Initialize an empty TensorArray to store the output
        output_tensor = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)

        for b in range(batch_size):
            temp_output = []
            for o in range(output_size):
                st_idx = (o + input_size) % input_size
                p = 0
                for i in range(f_):
                    new_i = (st_idx + i) % input_size
                    p += i_[b, new_i] * k_[o, new_i]

                temp_output.append(p)
            output_tensor = output_tensor.write(b, temp_output)

        # print("here 4")
        output_tensor = output_tensor.stack()
                
    return output_tensor, tape

def csc_fc(inputs, kernel, cc, cn, cf, cs):
    with tf.compat.v1.get_default_graph().gradient_override_map({"MatMul": "CSC_FC"}):
        result, _ = csc_fc_kernel(inputs, kernel, cc, cn, cf, cs)
    return result

@register_keras_serializable(package='Custom', name='CSC_FC')
class CSC_FC(Layer):
    def __init__(
        self,
        units,
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

        self.CustomMatrix = np.full((self.CSC_C, self.CSC_N), 0)
        for f in range(0, self.CSC_F):
            self.CustomMatrix[0, f * self.CSC_S] = 1
        for c in range(1, self.CSC_C):
            for n in range(0, self.CSC_N):
                self.CustomMatrix[c, n] = self.CustomMatrix[c - 1, (n - 1) % self.CSC_N]
        self.CustomMatrix = tf.convert_to_tensor(self.CustomMatrix, dtype=tf.float32)

        assert self.kernel.shape == self.CustomMatrix.shape, "Filter multiplied with a different shape array " + str(
            self.CustomMatrix.shape) + " " + str(self.kernel.shape)

        # self.CustomMatrix = np.full((self.CSC_N, self.CSC_C), 0)
        # for f in range(0, self.CSC_F):
        #     self.CustomMatrix[0, f * self.CSC_S] = 1
        # for n in range(1, self.CSC_N):
        #     for c in range(0, self.CSC_C):
        #         self.CustomMatrix[n, c] = self.CustomMatrix[n - 1, (c - 1) % self.CSC_C]
        # # self.CustomMatrix = tf.convert_to_tensor(self.CustomMatrix, dtype=tf.float32)

        # assert self.kernel.shape == self.CustomMatrix.shape, "Filter multiplied with a different shape array\n" + "CustomMatrix Shape: " + str(
        #     self.CustomMatrix.shape) + " \nkernel Shape: " + str(self.kernel.shape)


    @custom_gradient.custom_gradient
    def call_with_custom_grad(self, inputs, kernel, bias):
        # Implement the custom op with the gradient
        output, = tf.compat.v1.py_func(self.call_with_custom_op, [inputs, kernel, bias], [tf.float32])
        def grad(dy):
            return _csc_fc_tf_grad(output.op, dy)
        return output, grad

    def call_with_custom_op(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
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

        if rank == 2 or rank is None:
            if isinstance(inputs, tf.SparseTensor):
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
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
                # Call custom op for matrix multiplication
                outputs = csc_fc(inputs, self.kernel, self.CSC_C, self.CSC_N, self.CSC_F, self.CSC_S)
        else:
            raise ValueError("Input rank greater than 2 is not supported for this custom layer.")

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs


    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
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

        self.kernel.assign(tf.multiply(self.kernel, self.CustomMatrix))

        if rank == 2 or rank is None:
            if isinstance(inputs, tf.SparseTensor):
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
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
                # outputs = tf.matmul(a=inputs, b=self.kernel)
                outputs = csc_fc(inputs, self.kernel, self.CSC_C, self.CSC_N, self.CSC_F, self.CSC_S)
        else:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
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
            }
        )
        return config

