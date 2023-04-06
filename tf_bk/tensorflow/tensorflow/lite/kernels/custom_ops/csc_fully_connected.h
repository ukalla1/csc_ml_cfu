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

// The below note is to be re-written........
// Forward declares registrations for specific CSC-FC layer implementations. Do not
// include this header if you are fine with any FC implementation, include
// builtin_op_kernels.h instead. This implementation-specific registration is
// only available for CSC-FC, as these versions are explicitly tested and supported.

#ifndef TENSORFLOW_LITE_KERNELS_CSC_FC_H_
#define TENSORFLOW_LITE_KERNELS_CSC_FC_H_

#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/kernel_util.h"
// #include "tensorflow/lite/kernels/OpKernelContext.h"

namespace tflite{
    namespace ops{
        namespace custom{

            struct CSC_Params{
                int CSC_C;
                int CSC_N;
                int CSC_F;
                int CSC_S;
            };

            // class CSC_FC : public OpKernel {
            class CSC_FC {

                public:
                    // static TfLiteRegistration* Register_CSC_FC();
                    static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
                    static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);
                    static TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node);

                // private:
                //     CSC_Params params_;
            }; 

            extern "C" TfLiteRegistration* TFL_RegisterCSC_FC();

        } //namespace custom
    } //namespace ops
} //namespace tflite


#endif  // TENSORFLOW_LITE_KERNELS_FC_H_
