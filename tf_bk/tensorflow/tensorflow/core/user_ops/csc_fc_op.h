// #ifndef CSC_FC_OP_H_
// #define CSC_FC_OP_H_

// #include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/shape_inference.h"
// #include <iostream>

// using namespace tensorflow;

// // Register the custom CSC_FC op
// REGISTER_OP("CscFc")
//     .Input("inputs: float")
//     .Input("kernel: float")
//     .Input("bias: float")
//     .Input("csc_c: int32")
//     .Input("csc_n: int32")
//     .Input("csc_f: int32")
//     .Input("csc_s: int32")
//     .Output("output: float")
//     .SetShapeFn([](shape_inference::InferenceContext* c) {
//       shape_inference::ShapeHandle input_shape;
//       shape_inference::ShapeHandle kernel_shape;
//       shape_inference::ShapeHandle bias_shape;
//       shape_inference::DimensionHandle batch_size_dim;
//       shape_inference::DimensionHandle output_channels_dim;

//       TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
//       TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
//       TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bias_shape));

//       batch_size_dim = c->Dim(input_shape, 0);
//       output_channels_dim = c->Dim(kernel_shape, 1);

//       c->set_output(0, c->MakeShape({batch_size_dim, output_channels_dim}));
//       return OkStatus();
//     });

// // Register the custom CSC_FC_GRAD op
// REGISTER_OP("CscFcGrad")
//     .Input("input: float")
//     .Input("kernel: float")
//     .Input("grad_output: float")
//     .Input("csc_c: int32")
//     .Input("csc_n: int32")
//     .Input("csc_f: int32")
//     .Input("csc_s: int32")
//     .Output("grad_input: float")
//     .Output("grad_kernel: float")
//     .Output("grad_bias: float")
//     .Doc(R"doc(
// Custom CSC fully connected gradient operation.
// )doc");

// class CscFcGradOp : public tensorflow::OpKernel {
//  public:
//   explicit CscFcGradOp(tensorflow::OpKernelConstruction* context);

//   void Compute(tensorflow::OpKernelContext* context) override;
// };

// #endif  // CSC_FC_OP_H_


#ifndef CSC_FC_OP_H_
#define CSC_FC_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;

// Register the custom CSC_FC op
REGISTER_OP("CscFc")
    .Input("inputs: float")
    .Input("kernel: float")
    .Input("bias: float")
    .Input("csc_c: int32")
    .Input("csc_n: int32")
    .Input("csc_f: int32")
    .Input("csc_s: int32")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      shape_inference::ShapeHandle kernel_shape;
      shape_inference::ShapeHandle bias_shape;
      shape_inference::DimensionHandle batch_size_dim;
      shape_inference::DimensionHandle output_channels_dim;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bias_shape));

      batch_size_dim = c->Dim(input_shape, 0);
      output_channels_dim = c->Dim(kernel_shape, 1);

      c->set_output(0, c->MakeShape({batch_size_dim, output_channels_dim}));
      return OkStatus();
    });

// Register the custom CSC_FC_GRAD op
REGISTER_OP("CscFcGrad")
    .Input("input: float")
    .Input("kernel: float")
    .Input("grad_output: float")
    .Input("csc_c: int32")
    .Input("csc_n: int32")
    .Input("csc_f: int32")
    .Input("csc_s: int32")
    .Output("grad_input: float")
    .Output("grad_kernel: float")
    .Output("grad_bias: float")
    .Doc(R"doc(
Custom CSC fully connected gradient operation.
)doc");

class CscFcGradOp : public tensorflow::OpKernel {
 public:
  explicit CscFcGradOp(tensorflow::OpKernelConstruction* context);

  void Compute(tensorflow::OpKernelContext* context) override;
};

#endif  // CSC_FC_OP_H_