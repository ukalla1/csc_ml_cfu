import tensorflow as tf
from tensorflow.lite.python import convert as _convert
from tensorflow.lite.python import lite_constants as _lite_constants
from tensorflow.lite.python.convert_saved_model import TFLiteSavedModelConverterV2

class CustomTFLiteConverter(TFLiteSavedModelConverterV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_opdefs = {}

    def register_custom_opdefs(self, custom_opdefs):
        self._custom_opdefs = custom_opdefs

    def _replace_tensors_with_buffers(self):
        for idx, op in enumerate(self._graph_def.op):
            if op.name in self._custom_opdefs:
                for k, v in self._custom_opdefs[op.name].items():
                    tensor = tf.constant(v, dtype=tf.int32)
                    buffer_name = f"{op.name}/{k}:0_const"
                    tensor = tf.identity(tensor, name=buffer_name)
                    self._graph_def.op[idx].input.append(tensor.name)

    def convert(self):
        # Store tensors as constant tensors
        self._replace_tensors_with_buffers()

        # Convert the model
        return super().convert()
