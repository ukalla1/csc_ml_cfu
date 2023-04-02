For tensorflow:  
1. To build the shared library run: 'bazel build //tensorflow/core/user_ops:all'  
    1.1 Noticing an undefined symbol error (symbol err goes here) is expected.  
        1.1.1 Might have to use tensorlfow 2.12.0.  

For tflite:  
1. To build the shared library run: 'bazel build //tensorflow/core/user_ops:all'  
