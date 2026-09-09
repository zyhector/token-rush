// Binding for the Token Rush port of Marlin (see marlin_bf16.cu). Apache-2.0.
#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace marlin_bf16 {
int marlin_bf16_cuda(const void* A, const void* B, void* C, const void* s, const void* mn, void* red,
                     int prob_m, int prob_n, int prob_k, void* workspace, int groupsize, int dev,
                     cudaStream_t stream, int thread_k, int thread_n, int sms, bool partial, int n_slots);
int partial_slots(int prob_n, int prob_k, int thread_k, int thread_n, int sms, int dev);
}

void mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, const torch::Tensor& s,
         const torch::Tensor& m, torch::Tensor& workspace, torch::Tensor& red,
         int thread_k = -1, int thread_n = -1, int sms = -1) {
  int prob_m = A.size(0);
  int prob_n = C.size(-1);
  int prob_k = A.size(1);
  int groupsize = prob_k / s.size(0);
  TORCH_CHECK(groupsize * s.size(0) == prob_k, "k=", prob_k, " not compatible with ", s.size(0), " groups");
  TORCH_CHECK(workspace.numel() >= prob_n / 128, "workspace too small");
  bool partial = C.dtype() == torch::kFloat32;      // C is then the [n_slots, m, n] fp32 partial buffer
  int n_slots = partial ? C.size(0) : 0;
  if (partial) {
    TORCH_CHECK(C.dim() == 3 && C.size(1) == prob_m && n_slots >= marlin_bf16::partial_slots(prob_n, prob_k, thread_k, thread_n, sms, A.get_device()),
                "partial buffer must be [slots, m, n] with slots >= partial_slots()");
  } else {
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16 (or an fp32 [slots, m, n] partial buffer)");
    TORCH_CHECK(red.numel() >= (int64_t) 16 * prob_n, "fp32 reduce buffer too small");
  }
  TORCH_CHECK(A.dtype() == torch::kBFloat16 && s.dtype() == torch::kBFloat16 &&
              m.dtype() == torch::kBFloat16 && red.dtype() == torch::kFloat32, "dtypes");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() && s.is_contiguous() && m.is_contiguous());
  int dev = A.get_device();
  int err = marlin_bf16::marlin_bf16_cuda(
    A.data_ptr(), B.data_ptr(), partial ? nullptr : C.data_ptr(), s.data_ptr(), m.data_ptr(),
    partial ? C.data_ptr() : red.data_ptr(),
    prob_m, prob_n, prob_k, workspace.data_ptr(), groupsize, dev,
    at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms, partial, n_slots);
  TORCH_CHECK(err != 1, "problem shape (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ") not supported: "
              "need m <= 16, n % ", thread_n, " == 0, k % ", thread_k, " == 0");
  TORCH_CHECK(err != 2, "no kernel for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize);
}

int partial_slots(int n, int k, int thread_k = -1, int thread_n = -1, int sms = -1) {
  return marlin_bf16::partial_slots(n, k, thread_k, thread_n, sms, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("partial_slots", &partial_slots, "fp32 slots a partial-mode launch writes",
          py::arg("n"), py::arg("k"), py::arg("thread_k") = -1, py::arg("thread_n") = -1, py::arg("sms") = -1);
  mod.def("mul", &mul, "bf16 x int4 (asymmetric, g128) matmul, Marlin layout",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("s"), py::arg("m"), py::arg("workspace"), py::arg("red"),
          py::arg("thread_k") = -1, py::arg("thread_n") = -1, py::arg("sms") = -1);
}
