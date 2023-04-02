# User ops.

#load(
#    "//tensorflow/core/platform:rules_cc.bzl",
#    "cc_library",
#)
#load(
#    "//tensorflow:tensorflow.bzl",
#    "tf_copts",
#)
#
#package(
#    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
#    default_visibility = [
#        "//tensorflow/core:__pkg__",
#    ],
#    licenses = ["notice"],
#)
#
# One target for all user ops.
#cc_library(
#    name = "user_ops_op_lib",
#    srcs = glob(["*.cc"]),
#    copts = tf_copts(),
#    linkstatic = 1,
#    visibility = ["//tensorflow/core:__pkg__"],
#    deps = ["//tensorflow/core:framework"],
#    alwayslink = 1,
#)

############################################################################# above orig


#load(
#    "//tensorflow/core/platform:rules_cc.bzl",
#    "cc_library",
#)
#load(
#    "//tensorflow:tensorflow.bzl",
#    "tf_copts",
#    "tf_custom_op_library",
#)
#
#package(
#    default_visibility = [
#        "//tensorflow/core:__pkg__",
#    ],
#    licenses = ["notice"],
#)
#
## One target for all user ops.
#cc_library(
#    name = "user_ops_op_lib",
#    srcs = glob(["*.cc"], exclude=["csc_fc_grad_op.cc"]),
#    hdrs = glob(["*.h"]),
#    copts = tf_copts(),
#    linkstatic = 1,
#    visibility = ["//tensorflow/core:__pkg__"],
#    deps = ["//tensorflow/core:framework"],
#    alwayslink = 1,
#)
#
#tf_custom_op_library(
#    name = "csc_fc_op.so",
#    srcs = [
#        "csc_fc_grad_op.cc",
#    ],
#    copts = ["-fPIC"],
#    deps = [
#        ":user_ops_op_lib",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#    ],
#)

##########################################################################################

#load(
#    "//tensorflow/core/platform:rules_cc.bzl",
#    "cc_library",
#)
#load(
#    "//tensorflow:tensorflow.bzl",
#    "tf_copts",
#    "tf_custom_op_library",
#)
#
#package(
#    default_visibility = [
#        "//tensorflow/core:__pkg__",
#    ],
#    licenses = ["notice"],
#)
#
# One target for all user ops.
#cc_library(
#    name = "user_ops_op_lib",
#    srcs = glob(["*.cc"], exclude=["csc_fc_grad_op.cc"]),
#    hdrs = glob(["*.h"]),
#    copts = tf_copts(),
#    linkstatic = 1,
#    visibility = ["//tensorflow/core:__pkg__"],
#    deps = ["//tensorflow/core:framework"],
#    alwayslink = 1,
#)
#
#tf_custom_op_library(
#    name = "csc_fc_op.so",
#    srcs = [
#        "csc_fc_op.cc",
#        "csc_fc_grad_op.cc",
#    ],
#    copts = ["-fPIC"],
#    deps = [
#        ":user_ops_op_lib",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#    ],
#)

################################################################################################# above worked

#load(
#    "//tensorflow/core/platform:rules_cc.bzl",
#    "cc_library",
#)
#load(
#    "//tensorflow:tensorflow.bzl",
#    "tf_copts",
#    "tf_custom_op_library",
#)
#
#package(
#    default_visibility = [
#        "//visibility:public",
#    ],
#    licenses = ["notice"],
#)
#
#cc_library(
#    name = "user_ops_op_lib",
#    srcs = glob(["*.cc"], exclude=["csc_fc_grad_op.cc"]),
#    hdrs = glob(["*.h"]),
#    copts = tf_copts(),
#    linkstatic = 1,
#    deps = [
#        "//tensorflow/core:framework",
#    ],
#    alwayslink = 1,
#)
#
#tf_custom_op_library(
#    name = "csc_fc_op.so",
#    srcs = [
#        "csc_fc_op.cc",
#        "csc_fc_grad_op.cc",
#    ],
#    copts = ["-fPIC"],
#    deps = [
#        ":user_ops_op_lib",
#    ],
#)

############################################################################################# above worked

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_custom_op_library",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
#    copts = tf_copts() + ["-I/usr/include/python3.10", "-I./", "-I./numpy_include"],
    copts = tf_copts() + ["-I/usr/include/python3.10"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "csc_fc_op.so",
    copts = ["-fPIC"],
    deps = [
        ":user_ops_op_lib",
    ],
)