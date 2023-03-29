#load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
#
#load(":tflite_custom_ops.bzl", "tflite_copts")
#
#cc_library(
#    name = "CSC_FC",
#    srcs = [
#        "csc_fully_connected.cc",
#    ],
#    hdrs = [
#        "csc_fully_connected.h",
#    ],
#    copts = tflite_copts(),
#    deps = [
#        "//tensorflow/lite:framework",
#        "//tensorflow/lite/kernels:builtin_ops",
#        "//tensorflow/lite/c:common",
#    ],
#    visibility = ["//visibility:public"],
#)
#

cc_library(
    name = "libCSC_FC",
    srcs = [
        "csc_fully_connected.cc",
    ],
    hdrs = [
        "csc_fully_connected.h",
    ],
    deps = [
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
    ],
    visibility = ["//visibility:public"],
)