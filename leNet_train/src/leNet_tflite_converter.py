# import tensorflow as tf
# import keras
# from keras.models import Model
# import os
# import matplotlib as plt

# from csc_dense_k2 import CSC_FC

# import ctypes

# class TfLiteRegistration(ctypes.Structure):
#     _fields_ = [
#         ('init', ctypes.c_void_p),
#         ('free', ctypes.c_void_p),
#         ('prepare', ctypes.c_void_p),
#         ('invoke', ctypes.c_void_p),
#         ('profiling_string', ctypes.c_void_p),
#         ('builtin_code', ctypes.c_int),
#         ('custom_name', ctypes.c_char_p),
#         ('version', ctypes.c_int)
#     ]

# def get_custom_op_registration():
#     # Load the shared library
#     lib_path = "/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so"
#     lib = ctypes.pydll.LoadLibrary(lib_path)

#     # Get the registration function from the shared library
#     lib.CSC_FC.restype = ctypes.POINTER(TfLiteRegistration)
#     registration_ptr = lib.CSC_FC()

#     # Dereference the pointer and return the TfLiteRegistration object
#     return registration_ptr.contents


# def load_model(path):
# 	m = tf.keras.models.load_model(path, custom_objects={'CSC_FC': CSC_FC})
# 	custom_objects = {"CSC_FC": CustomLayer}
# 	with keras.utils.custom_object_scope(custom_objects):
# 		m = keras.Model.from_config(config)
# 	return m

# def main():
# 	model_pth = '../bin/models/'
# 	lib_path = "/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so"

# 	lib = ctypes.CDLL(lib_path)
# 	RegisterCSC_FC = lib.RegisterCSC_FC
# 	RegisterCSC_FC.restype = ctypes.POINTER(ctypes.c_void_p)
# 	# model = load_model(model_pth)
# 	m = tf.saved_model.load(model_pth)

# 	concrete_func = m.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# 	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# 	# custom_op_mapping = {
# 	# "CSC_FC": get_custom_op_registration()
# 	# }

# 	print("Success!!!")

# if __name__ == "__main__":
# 	main()

###################################################################################################################

# import tensorflow as tf
# from tensorflow.lite.python import convert
# import ctypes

# def load_custom_op_lib():
#     return ctypes.CDLL("/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so")

# def get_custom_op_registration():
#     custom_op_lib = load_custom_op_lib()
#     custom_op_lib.TFL_RegisterCSC_FC.restype = ctypes.c_void_p
#     return custom_op_lib.TFL_RegisterCSC_FC()

# # Load your SavedModel
# path = "../bin/models/"
# loaded_model = tf.saved_model.load(path)

# # Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model(path)
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]

# converter.experimental_custom_register_opdefs = [get_custom_op_registration().contents]

# tflite_model = converter.convert()

# # Save the TFLite model
# with open("../bin/models/test_tflite/converted_model.tflite", "wb") as f:
#     f.write(tflite_model)

###################################################################################################################

# import ctypes
# import os
# import tensorflow as tf
# from tensorflow.lite.python import convert

# def load_custom_op_lib():
#     tflite_lib_path = "/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so"
#     custom_op_lib_path = "/home/uttej/work/ra/tf/tensorflow/bazel-bin/tensorflow/lite/kernels/custom_ops/liblibCSC_FC.so"
    
#     tflite_lib = ctypes.pydll.LoadLibrary(tflite_lib_path)
#     custom_op_lib = ctypes.pydll.LoadLibrary(custom_op_lib_path)

#     return custom_op_lib

# def get_custom_op_registration():
#     custom_op_lib = load_custom_op_lib()
#     custom_op_lib.TFL_RegisterCSC_FC.restype = ctypes.c_void_p
#     return custom_op_lib.TFL_RegisterCSC_FC()

# # Load your SavedModel
# path = "../bin/models/"
# loaded_model = tf.saved_model.load(path)

# converter = tf.lite.TFLiteConverter.from_saved_model(path)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, get_custom_op_registration()]

# tflite_model = converter.convert()

# # Save the TFLite model
# with open("../bin/models/test_tflite/converted_model.tflite", "wb") as f:
#     f.write(tflite_model)

###################################################################################################################

# import tensorflow as tf
# from tensorflow import lite
# # from tensorflow.lite.experimental import OpHint

# # Load your saved model with the custom operation
# saved_model_path = "../bin/models/"

# # Create the converter object
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# # Register your custom operation
# # def CSC_FC(converter: lite.ConverterOps):
# #     return lite.OpHint(
# #         "CSC_FC",
# #         custom_opcode=1,
# #         inputs=[converter.inputs[0], converter.inputs[1], converter.inputs[2]],
# #         outputs=[converter.outputs[0]],
# #     )

# def CSC_FC(x, weights, biases, CSC_C, CSC_N, CSC_F, CSC_S, name=None):
#     csc_fc_params = {
#         'CSC_C': tf.constant(CSC_C, dtype=tf.int32),
#         'CSC_N': tf.constant(CSC_N, dtype=tf.int32),
#         'CSC_F': tf.constant(CSC_F, dtype=tf.int32),
#         'CSC_S': tf.constant(CSC_S, dtype=tf.int32),
#     }
#     hint = OpHint('CSC_FC', trainable_variables={'weights': weights, 'biases': biases}, inputs=[x], outputs=['output'], attrs=csc_fc_params)
#     return hint

# # Add the custom operation to the converter
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS,
# ]
# converter.representative_dataset = None
# converter.experimental_op_hints = [CSC_FC]

# # Convert the model
# tflite_model = converter.convert()

# # Save the converted model
# with open("converted_model.tflite", "wb") as f:
#     f.write(tflite_model)

###################################################################################################################

# import tensorflow as tf
# from tensorflow.lite.python import schema_py_generated as schema
# from tensorflow import lite


# def map_function(op):
#     if op.type == 'CSC_FC':
#         custom_op = tf.lite.OperatorDef()
#         custom_op.opcode_index = 0
#         custom_op.inputs.extend([0, 1, 2, 3, 4, 5, 6])
#         custom_op.outputs.append(7)
#         return custom_op
#     return None


# # Load your saved model with the custom operation
# saved_model_path = "../bin/models/"

# # Create the converter object
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 'CSC_FC']

# # Convert the model
# tflite_model = converter.convert()

# # Load the FlatBuffer model
# model = schema.Model.GetRootAsModel(tflite_model, 0)

# # Replace tensors with buffers
# model = tf.lite.experimental.replace_tensors_with_buffers(model, map_function)

# # Save the converted model
# with open('converted_model.tflite', 'wb') as f:
#     f.write(model)

###################################################################################################################

# import numpy as np
# import tensorflow as tf
# from tensorflow.lite.python import schema_py_generated as schema
# from flatbuffers import Builder

# # Load your saved model with the custom operation
# saved_model_dir = "../bin/models/"

# # Create a converter
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# # Set the custom ops
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 'CSC_FC']

# # Convert the model
# tflite_model = converter.convert()

# # Load the FlatBuffer model
# model = schema.Model.GetRootAsModel(tflite_model, 0)

# # Create a new FlatBuffer builder
# builder = Builder(1024)

# # Find the index of the custom op in the opcodes table
# custom_op_index = -1
# for i in range(model.OperatorCodesLength()):
#     op_code = model.OperatorCodes(i)
#     if op_code.CustomCode() == b'CSC_FC':
#         custom_op_index = i
#         break

# if custom_op_index == -1:
#     raise ValueError('Custom op not found in the opcodes table')

# # Function to replace the built-in ops with the custom op
# def replace_ops(subgraph):
#     for i in range(subgraph.OperatorsLength()):
#         op = subgraph.Operators(i)

#         if op.OpCodeIndex() == custom_op_index:
#             # Modify the op's inputs and outputs
#             op.InputsAsNumpy()[3:7] = np.arange(3, 7, dtype=np.int32)
#             op.OutputsAsNumpy()[0] = 7

# # Modify the main subgraph
# replace_ops(model.Subgraphs(0))

# # Serialize the modified model
# model_bytes = model.Pack(builder)
# builder.Finish(model_bytes)

# # Save the converted model
# with open('converted_model.tflite', 'wb') as f:
#     f.write(builder.Output())

###################################################################################################################

# import numpy as np
# import tensorflow as tf
# from tensorflow.lite.python import schema_py_generated as schema
# from flatbuffers import Builder
# from custom_tflite_converter import CustomTFLiteConverter

# # Load your saved model with the custom operation
# saved_model_dir = "../bin/models/"

# # Create a converter
# converter = CustomTFLiteConverter.from_saved_model(saved_model_dir)

# # Register the custom op definitions
# converter.register_custom_opdefs({'CSC_FC': {'CSC_C': 0, 'CSC_N': 0, 'CSC_F': 0, 'CSC_S': 0}})

# # Set the custom ops
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 'CSC_FC']

# # Convert the model
# tflite_model = converter.convert()

# # Save the TFLite model
# with open("custom_csc_fc_model.tflite", "wb") as f:
#     f.write(tflite_model)

###################################################################################################################

# import tensorflow as tf

# def register_custom_opdefs(graph_def, custom_opdefs):
#     for idx, op in enumerate(graph_def.op):
#         if op.name in custom_opdefs:
#             for k, v in custom_opdefs[op.name].items():
#                 tensor = tf.constant(v, dtype=tf.int32)
#                 buffer_name = f"{op.name}/{k}:0_const"
#                 tensor = tf.identity(tensor, name=buffer_name)
#                 graph_def.op[idx].input.append(tensor.name)


# def convert_model(saved_model_dir, custom_opdefs):
#     converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

#     # Save the original graph_def
#     original_graph_def = converter.graph_def

#     # Modify the graph_def
#     modified_graph_def = tf.compat.v1.GraphDef()
#     modified_graph_def.CopyFrom(original_graph_def)
#     register_custom_opdefs(modified_graph_def, custom_opdefs)

#     # Set the converter's graph_def to the modified one
#     converter.graph_def = modified_graph_def

#     # Convert the model
#     tflite_model = converter.convert()

#     return tflite_model


# # Load the saved model
# saved_model_dir = "../bin/models/"
# custom_opdefs = {'CSC_FC': {'CSC_C': 0, 'CSC_N': 0, 'CSC_F': 0, 'CSC_S': 0}}

# # Convert the model
# tflite_model = convert_model(saved_model_dir, custom_opdefs)

# # Save the converted model
# with open("converted_model.tflite", "wb") as f:
#     f.write(tflite_model)

###################################################################################################################

import tensorflow as tf

# Load your SavedModel
saved_model_dir = "../bin/models/"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable the custom op in the converter
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow Select ops
]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)

