#include "tensorflow/core/user_ops/csc_fc_op.h"
// #include "tensorflow/core/user_ops/pyo_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <Python.h>
// #include "numpy/arrayobject.h"
// #include "tensorflow/core/util/pyo_helper.h"

using namespace tensorflow;

// Helper function to perform the custom CSC_FC operation
template <typename T>
void CscFcKernel(const Tensor& input_tensor, const Tensor& kernel_tensor, const Tensor& bias_tensor,
                 const int csc_c, const int csc_n, const int csc_f, const int csc_s, Tensor* output_tensor) {

  // std::cout << "CscFcKernel called" << std::endl;

  auto i_ = input_tensor.tensor<T, 2>();
  auto k_ = kernel_tensor.tensor<T, 2>();
  auto b_ = bias_tensor.tensor<T, 1>();
  auto o_ = output_tensor->tensor<T, 2>();

  int batch_size = input_tensor.dim_size(0);
  int input_size = input_tensor.dim_size(1);
  int output_size = kernel_tensor.dim_size(1);

  for (int b = 0; b < batch_size; ++b) {
    int st = 0;
    for (int o = 0; o < output_size; ++o) {
      int st_idx = (o + input_size) % input_size;
      T p = b_(o); // Initialize with bias value
      for (int i = 0; i < csc_f; ++i) {
        int new_i = (st_idx + i) % input_size;
        p += i_(b, new_i) * k_(o, new_i);
      }
      o_(b, o) = p;
    }
  }
}

// Helper function to perform the custom gradient operation
template <typename T>
void CscFcGradKernel(const Tensor& input_tensor, const Tensor& kernel_tensor, const Tensor& grad_output_tensor,
                     const int csc_c, const int csc_n, const int csc_f, const int csc_s,
                     Tensor* grad_input_tensor, Tensor* grad_kernel_tensor, Tensor* grad_bias_tensor) {

  // std::cout << "CscFcGradKernel called" << std::endl;

  auto i_ = input_tensor.tensor<T, 2>();
  auto k_ = kernel_tensor.tensor<T, 2>();
  auto g_o_ = grad_output_tensor.tensor<T, 2>();
  auto g_i_ = grad_input_tensor->tensor<T, 2>();
  auto g_k_ = grad_kernel_tensor->tensor<T, 2>();
  auto g_b_ = grad_bias_tensor->tensor<T, 1>();

  int batch_size = input_tensor.dim_size(0);
  int input_size = input_tensor.dim_size(1);
  int output_size = kernel_tensor.dim_size(1);

  // Gradient with respect to bias (dL/db)
  for (int o = 0; o < output_size; ++o) {
    T grad_bias = 0;
    for (int b = 0; b < batch_size; ++b) {
      grad_bias += g_o_(b, o);
    }
    g_b_(o) = grad_bias;
  }

  // Gradient with respect to kernel (dL/dk)
  for (int o = 0; o < output_size; ++o) {
    for (int i = 0; i < input_size; ++i) {
      T grad_kernel = 0;
      for (int b = 0; b < batch_size; ++b) {
        int st_idx = (o + input_size) % input_size;
        int new_i = (st_idx + i) % input_size;
        if (i < csc_f) {
          grad_kernel += g_o_(b, o) * i_(b, new_i);
        }
      }
      g_k_(o, i) = grad_kernel;
    }
  }

  // Gradient with respect to input (dL/di)
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < input_size; ++i) {
      T grad_input = 0;
      for (int o = 0; o < output_size; ++o) {
        int st_idx = (o + input_size) % input_size;
        int new_i = (st_idx + i) % input_size;
        if (i < csc_f) {
          grad_input += g_o_(b, o) * k_(o, new_i);
        }
      }
      g_i_(b, i) = grad_input;
    }
  }
}

// Forward operation implementation
class CscFcOp : public OpKernel {
 public:
  explicit CscFcOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    // std::cout << "CscFcOp::Compute called" << std::endl;

    // Get the input tensors
    const Tensor& input_tensor = context->input(0);
    const Tensor& kernel_tensor = context->input(1);
    const Tensor& bias_tensor = context->input(2);

    // Get the additional inputs
    int csc_c = context->input(3).scalar<int>()();
    int csc_n = context->input(4).scalar<int>()();
    int csc_f = context->input(5).scalar<int>()();
    int csc_s = context->input(6).scalar<int>()();

    // Calculate the output shape
    TensorShape output_shape = input_tensor.shape();
    output_shape.set_dim(1, kernel_tensor.dim_size(1));

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    // Call the helper function to perform the custom CSC_FC operation
    CscFcKernel<float>(input_tensor, kernel_tensor, bias_tensor, csc_c, csc_n, csc_f, csc_s, output_tensor);

    // std::cout << "CSC_FC_OP_CC included and kernel registered" << std::endl;
  }
};

CscFcGradOp::CscFcGradOp(OpKernelConstruction* context) : OpKernel(context) {}

void CscFcGradOp::Compute(OpKernelContext* context) {

    // std::cout << "CscFcGradOp::Compute called" << std::endl;

    // Get the input tensors
    const Tensor& input_tensor = context->input(0);
    const Tensor& kernel_tensor = context->input(1);
    const Tensor& grad_output_tensor = context->input(2);

        // Get the additional inputs
    int csc_c = context->input(3).scalar<int>()();
    int csc_n = context->input(4).scalar<int>()();
    int csc_f = context->input(5).scalar<int>()();
    int csc_s = context->input(6).scalar<int>()();

    // Allocate memory for output tensors
    Tensor* grad_input_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &grad_input_tensor));

    Tensor* grad_kernel_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, kernel_tensor.shape(), &grad_kernel_tensor));

    Tensor* grad_bias_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({kernel_tensor.dim_size(1)}), &grad_bias_tensor));

    // Call the kernel function
    CscFcGradKernel<float>(input_tensor, kernel_tensor, grad_output_tensor, csc_c, csc_n, csc_f, csc_s,
                            grad_input_tensor, grad_kernel_tensor, grad_bias_tensor);

    // std::cout << "CSC_FC_GRAD_OP_CC included and kernel registered" << std::endl;
  }

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_csc_fc_module(void) {
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "csc_fc_module",
        nullptr,
        -1,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    return PyModule_Create(&moduledef);
  }
#else
  PyMODINIT_FUNC initcsc_fc_module(void) {
    Py_InitModule("csc_fc_module", nullptr);
  }
#endif

REGISTER_KERNEL_BUILDER(Name("CscFc").Device(DEVICE_CPU), CscFcOp);
REGISTER_KERNEL_BUILDER(Name("CscFcGrad").Device(DEVICE_CPU), CscFcGradOp);