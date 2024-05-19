#include <ATen/ATen.h>
#include <torch/torch.h>

#include "grad_kernel.cuh"

int grad_cuda_forward(
    at::Tensor& input1,
    at::Tensor& output,
    int kernel_size, bool bilinear) {
      resample2d_kernel_forward(input1, output, s, t);
    return 1;
}

int grad_cuda_backward(
    at::Tensor& input1, 
    at::Tensor& gradOutput,
    int kernel_size, bool bilinear) {
        resample2d_kernel_backward(gradOutput, s, t);
    return 1;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &grad_cuda_forward, "grad forward (CUDA)");
  m.def("backward", &grad_cuda_backward, "grad backward (CUDA)");
}

