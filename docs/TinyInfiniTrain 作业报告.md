# TinyInfiniTrain 作业报告

## 一、test 通过截图

![](./assets/image.png)

## 二、作业步骤

> 将代码填入下面代码块中指定位置，并详细描述完成该作业的解决思路和遇到的问题。

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});

    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &input = grad_outputs[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});

    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}
```

#### 解决思路

照猫画虎，拿出 `input` 后，再拿出 `device, kernel` 最后直接进行分发调用

#### 遇到问题

N/A

### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    auto num_dim1 = input->Dims().size();
    auto num_dim2 = other->Dims().size();
    auto N = input->Dims()[num_dim1 - 2];
    auto M = input->Dims()[num_dim1 - 1];
    auto K = other->Dims()[num_dim2 - 1];
    CHECK_EQ(M, other->Dims()[num_dim2 - 2]); // check matrix multiply valid

    auto batch = num_dim1 == 3 ? input->Dims()[0] : 1;
    std::vector<int64_t> output_dims = batch == 1 ? std::vector<int64_t>{N, K} : std::vector<int64_t>{batch, N, K};

    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    for (auto b = 0; b < batch; b++) {
        output->EigenMatrix().block(b * N, 0, N, K)
            = input->EigenMatrix().block(b * N, 0, N, M) * other->EigenMatrix().block(b * M, 0, M, K);
    }

    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
            const std::shared_ptr<Tensor> &grad_output) {
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other->Dims(), other->Dtype(), other->GetDevice());

    auto batch = input->Dims().size() == 3 ? input->Dims()[0] : 1;
    auto N = input->Dims()[input->Dims().size() - 2];
    auto M = input->Dims()[input->Dims().size() - 1];
    auto K = other->Dims()[other->Dims().size() - 1];
    for (auto b = 0; b < batch; b++) {
        // compute grad for input: grad_input = grad_output * other^T
        grad_input->EigenMatrix().block(b * N, 0, N, M)
            = grad_output->EigenMatrix().block(b * N, 0, N, K) * other->EigenMatrix().block(b * M, 0, M, K).transpose();
        // compute grad for other: grad_other = input^T * grad_output
        grad_other->EigenMatrix().block(b * M, 0, M, K)
            = input->EigenMatrix().block(b * N, 0, N, M).transpose() * grad_output->EigenMatrix().block(b * N, 0, N, K);
    }

    return {grad_input, grad_other};
}
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    auto num_dim1 = input->Dims().size();
    auto num_dim2 = other->Dims().size();
    CHECK_EQ(num_dim1, num_dim2);

    auto M = input->Dims()[num_dim1 - 2];
    auto K = input->Dims()[num_dim1 - 1];
    auto N = other->Dims()[num_dim2 - 1];
    CHECK_EQ(K, other->Dims()[num_dim2 - 2]); // check matrix multiply valid
    auto bs = std::accumulate(input->Dims().rbegin() + 2, input->Dims().rend(), 1, std::multiplies<int64_t>{});
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    const float alpha = 1.0, beta = 0.0;

    // output^T = other^T * input^T for cuda GEMM
    // lhs = other^T: [..., N, K]
    // rhs = input^T: [..., K, M]
    // output^T: [..., N, M]
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    switch (input->Dtype()) {
    case DataType::kFLOAT32: {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, other->DataPtr(), CUDA_R_32F, N,
                                   N * K, input->DataPtr(), CUDA_R_32F, K, K * M, &beta, output->DataPtr(), CUDA_R_32F,
                                   N, M * N, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    case DataType::kBFLOAT16: {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, other->DataPtr(), CUDA_R_16BF, N,
                                   N * K, input->DataPtr(), CUDA_R_16BF, K, K * M, &beta, output->DataPtr(),
                                   CUDA_R_16BF, N, M * N, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    default: {
        LOG(FATAL) << "Unsupported data type in MatmulForward CUDA kernel.\n";
    }
    }

    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other->Dims(), other->Dtype(), other->GetDevice());

    /**
        grad_input = grad_output * other^T
                : [..., M, K]
        grad_other = input^T * grad_output
                : [..., K, N]
        grad_output: [..., M, N]
        other:     [..., K, N]
        input:     [..., M, K]
    */

    auto M = input->Dims()[input->Dims().size() - 2];
    auto K = input->Dims()[input->Dims().size() - 1];
    auto N = other->Dims()[other->Dims().size() - 1];
    CHECK_EQ(K, other->Dims()[other->Dims().size() - 2]); // check matrix multiply valid
    CHECK_EQ(M, grad_output->Dims()[grad_output->Dims().size() - 2]);
    CHECK_EQ(N, grad_output->Dims()[grad_output->Dims().size() - 1]);
    auto bs = std::accumulate(input->Dims().rbegin() + 2, input->Dims().rend(), 1, std::multiplies<int64_t>{});
    constexpr float alpha = 1.0, beta = 0.0;

    /**
        Column major:
            grad_input^T = other * grad_output^T
                grad_input^T = [..., K, M]
                other =    [..., K, N]
                grad_output^T = [..., N, M]
            grad_other^T = grad_output^T * input
                grad_other^T = [..., N, K]
                grad_output^T = [..., N, M]
                input =    [..., M, K]
    */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    switch (input->Dtype()) {
    case DataType::kFLOAT32: {
        // grad_input^T = other * grad_output^T
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, other->DataPtr(), CUDA_R_32F, N,
                                   K * N, grad_output->DataPtr(), CUDA_R_32F, N, N * M, &beta, grad_input->DataPtr(),
                                   CUDA_R_32F, K, M * K, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        // grad_other^T = grad_output^T * input
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, grad_output->DataPtr(),
                                   CUDA_R_32F, N, N * M, input->DataPtr(), CUDA_R_32F, K, M * K, &beta,
                                   grad_other->DataPtr(), CUDA_R_32F, N, K * N, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    case DataType::kBFLOAT16: {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, other->DataPtr(), CUDA_R_16BF, N,
                                   K * N, grad_output->DataPtr(), CUDA_R_16BF, N, N * M, &beta, grad_input->DataPtr(),
                                   CUDA_R_16BF, K, M * K, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        // grad_other^T = grad_output^T * input
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, grad_output->DataPtr(),
                                   CUDA_R_16BF, N, N * M, input->DataPtr(), CUDA_R_16BF, K, M * K, &beta,
                                   grad_other->DataPtr(), CUDA_R_16BF, N, K * N, bs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    default: {
        LOG(FATAL) << "Unsupported data type in MatmulBackward CUDA kernel.\n";
    }
    }

    return {grad_input, grad_other};
}
```

#### 解决思路

- CPU 侧的矩阵乘法使用 Eigen 库进行计算，使用 `.block()` 同时处理了二维和三维的情况
- CUDA 侧的矩阵乘法使用 cuBLAS 库进行计算，用 `cudaGemmStridedBatchedEx()` 同时处理多维的情况

#### 遇到问题

- CPU 侧基本没遇到什么问题，查了查 API 就行了
- CUDA 侧的主要问题是 column-major 下张量的维度设置错误，最后手推结合 GPT 解决。

### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：infini_train/src/kernels/cpu/accumulate_grad.cc

```c++
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
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
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
```

#### 解决思路

- CPU 端先提取出底层指针，然后直接循环原地修改
- CUDA 端也是先提取出底层指针，然后直接全部发射到 cuda kernel 上，每一个线程处理一个下标（且不会有 bank conflict）

#### 遇到问题

N/A

### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange) `,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    start = start < 0 ? start + dims_.size() : start;
    end = end < 0 ? end + dims_.size() : end;
    auto new_shape = std::vector<int64_t>(0);
    for (int i = 0; i < start; i++) { new_shape.push_back(dims_[i]); }
    int64_t shape = 1;
    for (int i = start; i <= end; i++) { shape *= dims_[i]; }
    new_shape.push_back(shape);
    for (int i = end + 1; i < dims_.size(); i++) { new_shape.push_back(dims_[i]); }

    return Contiguous()->View(new_shape);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    if (!grad_fn_) {
        return;
    }
    if (!gradient) {
        gradient = std::make_shared<Tensor>(std::vector<int64_t>{}, dtype_, GetDevice());
        gradient->Fill<float>(1);
    }

    grad_fn_->BackwardPartial(gradient, output_idx_);
}
```

#### 解决思路

- `Flatten()` 的实现主要是
    - 要考虑到 `start, end` $\lt 0$ 的情况，这点需要和 PyTorch 对齐
    - 然后就是直接复制 `dims_` 并计算 `[start, end]` 这几个维度的 dim 乘积即可，按顺序 `push_back()`
- `Backward()` 的实现主要是需要判断 `gradient` 是否为空，是的话，就当成标量进行创建；然后调用 `grad_fn_` 计算梯度 + 进行反向传播

#### 遇到问题

主要是需要理解 `Backward()` 函数的作用，然后考虑 `gradient/grad_fn_` 为空的 corner case.

### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    using FuncT = RetT (*)(ArgsT...);
    FuncT function = reinterpret_cast<FuncT>(func_ptr_);
    return function(args...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    CHECK(!key_to_kernel_map_.contains(key))
        << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
    key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static const bool _register_##kernel_name##__LINE__ = []() {                                                       \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                            \
        return true;                                                                                                   \
    }();
```

#### 解决思路

- `Call()` 的实现思路比较简单，就是做一个类型转换就可以直接调用了
- `Register()` 也比较直接，就是直接往 `map` 里 `emplace` 一下就好了
- `REGISTER_KERNEL()` 因为需要静态操作，所以需要用 `static const bool` + Lambda 函数静态执行。用 `do {} while(0)` 本质还是动态的。

#### 遇到问题

一开始的宏定义直接 `do {} while(0)` 简单粗暴，结果在 `.cu` 文件里碰到报错了，因为还是动态执行，得用 `static const bool` 配合 lambda 函数。

### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

完成以上所有作业，补齐训练框架的所有实现，理论上`TEST_F(GPT2TrainingTest, LogitsConsistency)`可以通过，在用例中判断比较预置的值和单步正向传播计算结果是否在误差允许范围内相等。

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    TinyShakespeareFile file;

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    // skip 8 bytes of magic and version
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    uint32_t magic = BytesToType<uint32_t>(header, 0);
    auto type_it = kTypeMap.find(static_cast<int>(magic));
    CHECK(type_it != kTypeMap.end()) << "Unsupported tinyshakespeare magic: " << magic;
    file.type = type_it->second;

    // get number of tokens and derive usable tokens/sequences
    uint32_t num_toks = BytesToType<uint32_t>(header, 8);
    size_t num_sequences = num_toks / sequence_length; // drop tail tokens that don't form a full sequence
    size_t usable_tokens = num_sequences * sequence_length;

    // read token data
    size_t type_size = kTypeToSize.at(file.type);
    size_t data_size = usable_tokens * type_size;

    file.dims = {static_cast<int64_t>(num_sequences), static_cast<int64_t>(sequence_length)};
    file.tensor = infini_train::Tensor(file.dims, DataType::kINT64);
    auto dst = reinterpret_cast<int64_t *>(file.tensor.DataPtr());

    if (file.type == TinyShakespeareType::kUINT16) {
        std::vector<uint16_t> buffer(usable_tokens);
        ifs.read(reinterpret_cast<char *>(buffer.data()), data_size);
        for (size_t i = 0; i < usable_tokens; ++i) { dst[i] = static_cast<int64_t>(buffer[i]); }
    } else if (file.type == TinyShakespeareType::kUINT32) {
        std::vector<uint32_t> buffer(usable_tokens);
        ifs.read(reinterpret_cast<char *>(buffer.data()), data_size);
        for (size_t i = 0; i < usable_tokens; ++i) { dst[i] = static_cast<int64_t>(buffer[i]); }
    } else {
        LOG(FATAL) << "Unsupported TinyShakespeareType.";
    }

    return file;
}

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
    token_size_in_bytes_(sizeof(int64_t)),
    sequence_size_in_bytes_(sequence_length * token_size_in_bytes_), num_samples_(text_file_.dims[0]) {}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    std::ifstream ifs(filepath);
    CHECK(ifs.is_open()) << "Failed to open file: " << filepath;

    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header, 0);

    uint32_t version = BytesToType<uint32_t>(header, 4);

    eot_token_ = 50256; // default to GPT-2 EOT

    vocab_size_ = BytesToType<uint32_t>(header, 8);

    token_table_.resize(vocab_size_);
    for (auto i = 0; i < vocab_size_; i++) {
        uint8_t length;
        ifs.read(reinterpret_cast<char *>(&length), sizeof(length));

        std::vector<char> token_bytes(length);
        ifs.read(token_bytes.data(), length);

        token_table_[i] = std::string(token_bytes.data(), length);
    }
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const { return token_table_.at(token_id); }
```

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    /* ...原代码... */
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        x = std::make_shared<infini_train::Tensor>(x->To(device));
        auto logits = model.Forward({x}).at(0);
        auto probs = nn::function::Softmax(logits, -1)->To(cpu);

        auto data = probs.DataPtr();
        auto vocab_size = probs.Dims()[2];
        float *P = reinterpret_cast<float *>(data) + (t - 1) * vocab_size;
        float coin = RandomF32(kRngState);
        int64_t next = SampleMult(P, vocab_size, coin);

        x = std::make_shared<infini_train::Tensor>(x_tensor.To(cpu));
        auto x_data = x->DataPtr();
        reinterpret_cast<int64_t *>(x_data)[t] = next;
        std::cout << Decode(next);
    }
    std::cout << std::endl;
}
```

#### 解决思路

- `Tokenizer` 的实现比较直接
    - `Decode()` 直接取即可
    - `GenerateText()` 按 Hint 的步骤来即可：先 `Forward`，然后取 logits，`Softmax` 计算概率，然后 `SampleMult` 进行采样，最后 `x_data[t]` 放回 tokens list，等待下一步 `Forward()`
- 数据读取就有点麻烦
    - 先读取 header 的信息
    - 然后计算需要读取的 tokens bytes size，读取完之后直接循环进行赋值
    - 需要注意的是，tokens 本身需要是 `kINT64`，但是存储可能是 `kUINT16` 或者 `kUINT32`，需要提前计算 byte size 然后手动类型转换。

#### 遇到问题

Tokenizer 初始化的坑在于其实 string 的长度和内容是写在 `.bin` 里的；此外 `.bin` 里的 `magic number` 和文件里的对应不上（文件里的 GPT2 Magic Number 定义也对不上……），所以这里直接写死了

数据读取那里，一开始总是遇到问题……后来发现 `.bin` 的数据似乎不对，有 `string` 是不全的。得对末尾的 `string` 截断……
