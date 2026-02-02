# TinyInfiniTrain 作业报告

## 一、test 通过截图

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
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
}
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
}
```

#### 解决思路



#### 遇到问题



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
    // =================================== 作业 ===================================
    // TODO：实现自动微分反向传播
    // 功能描述：1. 计算当前张量对叶子节点的梯度    2. 支持多输出场景的梯度累加
    // HINT: 
    // =================================== 作业 ===================================
}
```

#### 解决思路

- `Flatten()` 的实现主要是
    - 要考虑到 `start, end` $\lt 0$ 的情况，这点需要和 PyTorch 对齐
    - 然后就是直接复制 `dims_` 并计算 `[start, end]` 这几个维度的 dim 乘积即可，按顺序 `push_back()`

#### 遇到问题



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
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
}

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
}
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
    }
    std::cout << std::endl;
}
```

#### 解决思路



#### 遇到问题

