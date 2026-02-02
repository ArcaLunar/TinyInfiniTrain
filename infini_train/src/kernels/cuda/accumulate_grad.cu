#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include <cstdint>

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamAccGradKernel(float *g, float *m, float *v, float *p, float lr, float beta1, float beta2, float eps,
                                  float b1, float b2, int64_t t, int64_t numel) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numel) {
        m[tid] = beta1 * m[tid] + (1 - beta1) * g[tid];
        v[tid] = beta2 * v[tid] + (1 - beta2) * g[tid] * g[tid];

        float mhat = m[tid] / b1;
        float vhat = v[tid] / b2;

        p[tid] -= lr * mhat / (sqrtf(vhat) + eps);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    const float b1 = 1 - std::pow(beta1, (float)t);
    const float b2 = 1 - std::pow(beta2, (float)t);
    int64_t num_elements = grad->NumElements();

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    AdamAccGradKernel<<<blocks, threads>>>(static_cast<float *>(grad->DataPtr()), static_cast<float *>(m->DataPtr()),
                                           static_cast<float *>(v->DataPtr()), static_cast<float *>(param->DataPtr()),
                                           learning_rate, beta1, beta2, eps, b1, b2, t, num_elements);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
