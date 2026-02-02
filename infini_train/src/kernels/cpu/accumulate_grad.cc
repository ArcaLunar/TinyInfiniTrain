#include <cmath>
#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    const float b1 = 1.0 - std::pow(beta1, (float)t);
    const float b2 = 1.0 - std::pow(beta2, (float)t);

    auto *g = (float *)(grad->DataPtr());
    auto *m_ptr = (float *)(m->DataPtr());
    auto *v_ptr = (float *)(v->DataPtr());
    auto *p = (float *)(param->DataPtr());

    for (auto i = 0; i < grad->NumElements(); ++i) {
        m_ptr[i] = beta1 * m_ptr[i] + (1 - beta1) * g[i];
        v_ptr[i] = beta2 * v_ptr[i] + (1 - beta2) * g[i] * g[i];

        float m_hat = m_ptr[i] / b1;
        float v_hat = v_ptr[i] / b2;

        p[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
