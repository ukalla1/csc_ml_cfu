/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/kernels/custom_ops/csc_fully_connected.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"


namespace tflite{
    namespace ops{
        namespace custom{

            inline const TfLiteTensor* GetInput(TfLiteContext* context, TfLiteNode* node, int index) {
                return context->tensors + node->inputs->data[index];
            }

            inline TfLiteTensor* GetOutput(TfLiteContext* context, TfLiteNode* node, int index)  {
                return context->tensors + node->outputs->data[index];
            }
            
            TfLiteStatus CSC_FC::Prepare(TfLiteContext* context, TfLiteNode* node){
                
                // Get the custom parameters from the constant tensors
                const TfLiteTensor* CSC_C_tensor = tflite::GetInput(context, node, 3);
                const TfLiteTensor* CSC_N_tensor = tflite::GetInput(context, node, 4);
                const TfLiteTensor* CSC_F_tensor = tflite::GetInput(context, node, 5);
                const TfLiteTensor* CSC_S_tensor = tflite::GetInput(context, node, 6);

                CSC_Params params;

                params.CSC_C = *GetTensorData<int>(CSC_C_tensor);
                params.CSC_N = *GetTensorData<int>(CSC_N_tensor);
                params.CSC_F = *GetTensorData<int>(CSC_F_tensor);
                params.CSC_S = *GetTensorData<int>(CSC_S_tensor);

                // Prepare output tensor
                TfLiteTensor* output = tflite::GetOutput(context, node, 0);
                const RuntimeShape& input_shape = GetTensorShape(tflite::GetInput(context, node, 0));
                int num_dims = input_shape.DimensionsCount();
                TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
                for (int i = 0; i < num_dims; i++){
                    output_size->data[i] = input_shape.Dims(i);
                }
                output_size->data[output_size->size - 1] = GetTensorShape(tflite::GetInput(context, node, 1)).Dims(1);
                return context->ResizeTensor(context, output, output_size);
            }

            TfLiteStatus CSC_FC::Invoke(TfLiteContext* context, TfLiteNode* node){

                // Call the Eval method
                return Eval(context, node);
            }

            TfLiteStatus CSC_FC::Eval(TfLiteContext* context, TfLiteNode* node) {

                const TfLiteTensor* input = tflite::GetInput(context, node, 0);
                const TfLiteTensor* weights = tflite::GetInput(context, node, 1);
                const TfLiteTensor* biases = tflite::GetInput(context, node, 2);

                TfLiteTensor* output = tflite::GetOutput(context, node, 0);

                int batch_size = GetTensorShape(input).Dims(0);
                int input_depth = GetTensorShape(input).Dims(1);
                int output_depth = GetTensorShape(weights).Dims(1);

                // Retrieve the custom parameters from the user_data field of the node
                CSC_Params* params = reinterpret_cast<CSC_Params*>(node->user_data);

                // Access the custom parameters
                int CSC_C = params->CSC_C;
                int CSC_N = params->CSC_N;
                int CSC_F = params->CSC_F;
                int CSC_S = params->CSC_S;

                //implement the nested for loops here
                for (int b = 0; b < batch_size; ++b) {
                    for (int o = 0; o < output_depth; ++o) {
                        float sum = 0;
                        int start_idx = (o + input_depth) % input_depth;
                        for (int k = 0; k < CSC_F; ++k) {
                            int i = (start_idx + k) % input_depth;
                            sum += GetTensorData<float>(input)[b * input_depth + i] * GetTensorData<float>(weights)[i * output_depth + o];
                        }
                        sum += GetTensorData<float>(biases)[o];

                        GetTensorData<float>(output)[b * output_depth + o] = sum;
                    }
                }

                return kTfLiteOk;
            }

            extern "C" TFL_CAPI_EXPORT TfLiteRegistration* TFL_RegisterCSC_FC() {
                static TfLiteRegistration r = {nullptr, nullptr, CSC_FC::Prepare, CSC_FC::Invoke, "CscFc"};
                return &r;
            }


        } //namespace custom
    } //namespace ops
} //namespace tflite