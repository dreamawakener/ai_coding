#include "tensor_lib.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// CUDA 线程块大小设定
#define kCudaThreadsNum 256

// 计算所需的 Block 数量
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

// CUDA 错误检查辅助函数
inline void checkCuda(cudaError_t e, const char* msg = "") {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(e) << std::endl;
        throw std::runtime_error(cudaGetErrorString(e));
    }
}

// 全局 cuBLAS 句柄（用于矩阵乘法）
static cublasHandle_t g_cublas = nullptr;
static bool is_cuda_initialized = false;

// 初始化 cuBLAS
static void init_cuda_resources() {
    if (!is_cuda_initialized) {
        cublasCreate(&g_cublas);
        is_cuda_initialized = true;
    }
}

// ---------------------------------------------------------
// KERNELS (CUDA 核函数)
// ---------------------------------------------------------

// ReLU 前向传播：y = max(0, x)
__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx] > 0 ? input[idx] : 0.0f;
}

// ReLU 反向传播：如果原输入 > 0，梯度透传，否则梯度为 0
__global__ void relu_backward_kernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
}

// Sigmoid 前向：y = 1 / (1 + exp(-x))
__global__ void sigmoid_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

// Sigmoid 反向：dx = dy * y * (1 - y)
__global__ void sigmoid_backward_kernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = 1.0f / (1.0f + expf(-input[idx]));
        grad_input[idx] = grad_output[idx] * y * (1.0f - y);
    }
}

// 偏置加法：支持广播机制，将 bias 加到输出 tensor 上
// Cout 是通道数，col_cols 是每个通道的像素数 (H*W)
__global__ void bias_add_kernel(float* out, const float* bias, int N, int Cout, int col_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * col_cols;
    // Grid-Stride Loop 模式，处理数据量超过线程总数的情况
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int tmp = i / col_cols;
        int co = tmp % Cout; // 计算当前处于哪个通道
        out[i] += bias[co];
    }
}

// 全连接层偏置反向传播：db = sum(dY, axis=0)
// 使用 atomicAdd 因为多个线程可能同时写入同一个 bias 的梯度
__global__ void fc_bias_backward_kernel(const float* dY, float* db, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int c = i % C;
        atomicAdd(&db[c], dY[i]);
    }
}

// 卷积层偏置反向传播：对 Batch 和 H*W 维度求和
__global__ void conv2d_bias_backward_kernel(const float* dY, float* db, int N, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int tmp = i / HW;
        int c = tmp % C;
        atomicAdd(&db[c], dY[i]);
    }
}

// Softmax 前向计算：使用了 Shared Memory 进行 Block 内规约（Reduction）以提高性能
__global__ void softmax_forward_kernel(const float* __restrict__ logits, float* __restrict__ probs, int N, int C) {
    int n = blockIdx.x; // 每个 Block 处理一个样本 (Batch中的一行)
    if (n >= N) return;
    
    // 共享内存，用于存储当前 Block 的中间最大值和求和
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // 1. 寻找最大值 (为了数值稳定性，计算 exp(x - max))
    float local_max = -1e30f;
    for (int i = tid; i < C; i += blockDim.x) {
        float v = logits[n * (long)C + i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    // 树状规约求最大值
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float mx = sdata[0];
    __syncthreads();

    // 2. 计算 exp 并求和
    float local_sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float e = expf(logits[n * (long)C + i] - mx);
        probs[n * (long)C + i] = e; // 暂时存储 exp 值
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    // 树状规约求和
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    // 3. 归一化得到概率
    for (int i = tid; i < C; i += blockDim.x) {
        probs[n * (long)C + i] /= sum;
    }
}

// 计算 Cross Entropy Loss
__global__ void cross_entropy_loss_kernel(const float* probs, const int* labels, float* out_losses, int N, int C) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lb = labels[n];
    float p = probs[n * (long)C + lb];
    // 加一个小量 1e-12 防止 log(0)
    out_losses[n] = -logf(fmaxf(p, 1e-12f));
}

// Cross Entropy + Softmax 的反向传播梯度：grad = (probs - 1_target) / N
__global__ void cross_entropy_backward_kernel(const float* probs, const int* labels, float* grad_logits, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int n = i / C;
        int c = i % C;
        float g = probs[i];
        if (labels[n] == c) g -= 1.0f; // 只在正确类别处减 1
        grad_logits[i] = g / float(N); // 除以 BatchSize 做平均
    }
}

// 矩阵转置：用于配合 cuBLAS 要求的列主序或调整维度顺序
__global__ void transpose_kernel(const float* src, float* dst, int R, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = R * C;
    if (idx >= total) return;
    int r = idx / C;
    int c = idx % C;
    dst[c * (size_t)R + r] = src[r * (size_t)C + c];
}

// Im2Col (Image to Column)：将图片卷积窗口展平成列，以便将卷积转化为矩阵乘法
__global__ void im2col_kernel_batch(const float* data_im, int N, int C, int H, int W,
                                    int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w,
                                    int H_out, int W_out, float* data_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K = C * kh * kw; // 卷积核体积（展平后每列的元素数）
    long total = (long)N * K * (H_out * W_out);
    for (long index = idx; index < total; index += blockDim.x * gridDim.x) {
        // 计算当前索引对应的输出位置、卷积核位置等
        int tmp = index;
        int out_pos = tmp % (H_out * W_out); tmp /= (H_out * W_out);
        int k = tmp % K; tmp /= K;
        int n = tmp;
        int c = k / (kh * kw);
        int rem = k % (kh * kw);
        int kh_i = rem / kw;
        int kw_i = rem % kw;
        int h_out = out_pos / W_out;
        int w_out = out_pos % W_out;
        int im_row = h_out * stride_h - pad_h + kh_i;
        int im_col = w_out * stride_w - pad_w + kw_i;
        float val = 0.0f;
        // 边界检查（Padding）
        if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
            long src_idx = n * (long)C * H * W + c * (long)H * W + im_row * W + im_col;
            val = data_im[src_idx];
        }
        long dst_idx = n * (long)K * (H_out * W_out) + k * (long)(H_out * W_out) + out_pos;
        data_col[dst_idx] = val;
    }
}

// Col2Im：Im2Col 的逆操作，用于卷积层输入梯度计算 (atomicAdd 累加重叠区域)
__global__ void col2im_kernel_batch_atomic(const float* data_col, int N, int C, int H, int W,
                                            int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w,
                                            int H_out, int W_out, float* data_im_grad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K = C * kh * kw;
    long total = (long)N * K * (H_out * W_out);
    for (long index = idx; index < total; index += blockDim.x * gridDim.x) {
        int tmp = index;
        int out_pos = tmp % (H_out * W_out); tmp /= (H_out * W_out);
        int k = tmp % K; tmp /= K;
        int n = tmp;
        int c = k / (kh * kw);
        int rem = k % (kh * kw);
        int kh_i = rem / kw;
        int kw_i = rem % kw;
        int h_out = out_pos / W_out;
        int w_out = out_pos % W_out;
        int im_row = h_out * stride_h - pad_h + kh_i;
        int im_col = w_out * stride_w - pad_w + kw_i;
        if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
            long dst_idx = n * (long)C * H * W + c * (long)H * W + im_row * W + im_col;
            long src_idx = n * (long)K * (H_out * W_out) + k * (long)(H_out * W_out) + out_pos;
            // 原子加：因为一个输入像素可能参与多个输出位置的计算
            atomicAdd(&data_im_grad[dst_idx], data_col[src_idx]);
        }
    }
}

// MaxPool Forward：2x2 窗口，Stride 2
__global__ void maxpool2x2_forward_kernel(const float* input, int N, int C, int H, int W,
                                            int out_h, int out_w, float* output, int* mask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    for (int idx = index; idx < total; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        int pw = tmp % out_w; tmp /= out_w;
        int ph = tmp % out_h; tmp /= out_h;
        int c = tmp % C; tmp /= C;
        int n = tmp;
        int hstart = ph * 2;
        int wstart = pw * 2;
        float best = -1.0e30f;
        int best_idx = -1;
        // 遍历 2x2 区域找最大值
        for (int i=0;i<2;++i) for (int j=0;j<2;++j) {
            int r = hstart + i;
            int col = wstart + j;
            if (r < H && col < W) {
                int in_idx = n * (long)C * H * W + c * (long)H * W + r * W + col;
                float v = input[in_idx];
                if (v > best) { best = v; best_idx = in_idx; }
            }
        }
        output[idx] = best;
        mask[idx] = best_idx; // 记录最大值索引供反向传播使用
    }
}

// MaxPool Backward：将梯度传回最大值所在的位置
__global__ void maxpool2x2_backward_kernel(const float* grad_out, const int* mask, int N, int C, int H, int W,
                                            int out_h, int out_w, float* grad_in) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    for (int idx = index; idx < total; idx += blockDim.x * gridDim.x) {
        int in_pos = mask[idx];
        if (in_pos >= 0) atomicAdd(&grad_in[in_pos], grad_out[idx]);
    }
}

// SGD + Momentum 更新
__global__ void sgd_momentum_kernel(float* param, const float* grad, float* velocity, float lr, float momentum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        float v = velocity[i];
        float p = param[i];

        v = momentum * v + g; // 更新动量
        p = p - lr * v;       // 更新参数

        velocity[i] = v;
        param[i] = p;
    }
}

// 简化的 Softmax + CrossEntropy 联合 Kernel
__global__ void softmax_ce_kernel(  const float* logits, float* probs, const int* labels,
                                    float* loss_acc, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        // 计算当前样本的 Softmax
        float max_val = -1e20f;
        for(int c=0; c<C; ++c) max_val = fmaxf(max_val, logits[n*C + c]);

        float sum = 0.0f;
        for(int c=0; c<C; ++c) {
            float val = expf(logits[n*C + c] - max_val);
            probs[n*C + c] = val;
            sum += val;
        }

        for(int c=0; c<C; ++c) probs[n*C + c] /= sum;

        // 累加 Loss
        int label = labels[n];
        float p = fmaxf(probs[n*C + label], 1e-10f);
        atomicAdd(loss_acc, -logf(p));
    }
}

// 交叉熵反向传播：简化版
__global__ void ce_backward_kernel(float* grad, const float* probs, const int* labels, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int n = idx / C, c = idx % C;
        float target = (labels[n] == c) ? 1.0f : 0.0f;
        grad[idx] = (probs[idx] - target) / N; // 核心：梯度需除以 N
    }
}

// 简单的向量加法 Kernel
__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------
// TENSOR IMPLEMENTATION (Tensor 类方法实现)
// ---------------------------------------------------------

Tensor::Tensor() : data_(nullptr), shape_(), device_(Device::CPU), size_(0) {}

// 构造函数：根据形状分配内存
Tensor::Tensor(const std::vector<int>& shape, Device device) : shape_(shape), device_(device) {
    size_ = 1;
    for (int n : shape) size_ *= n;
    if (size_ == 0) { data_.reset(); return; }

    if (device == Device::CPU) {
        data_ = std::shared_ptr<float>(new float[size_], std::default_delete<float[]>());
    } else {
        init_cuda_resources();
        float* ptr;
        checkCuda(cudaMalloc(&ptr, size_ * sizeof(float)));
        // 使用 lambda 表达式定义自定义删除器，当 shared_ptr 引用计数为 0 时调用 cudaFree
        data_ = std::shared_ptr<float>(ptr, [](float* p){ if(p) cudaFree(p); });
    }
}

// 构造函数：从现有 vector 数据初始化
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data, Device device) : shape_(shape), device_(device) {
    size_ = 1; for (int n : shape) size_ *= n;

    if (device == Device::CPU) {
        data_ = std::shared_ptr<float>(new float[size_], std::default_delete<float[]>());
        memcpy(data_.get(), data.data(), size_ * sizeof(float));
    } else {
        init_cuda_resources();
        float* ptr;
        checkCuda(cudaMalloc(&ptr, size_ * sizeof(float)));
        // 将数据从 Host 拷贝到 Device
        checkCuda(cudaMemcpy(ptr, data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
        data_ = std::shared_ptr<float>(ptr, [](float* p){ if(p) cudaFree(p); });
    }
}

// 深拷贝
Tensor Tensor::clone() const {
    Tensor res(shape_, device_);
    if (size_ > 0) {
        if (device_ == Device::CPU) memcpy(res.data_.get(), data_.get(), size_ * sizeof(float));
        else checkCuda(cudaMemcpy(res.data_.get(), data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return res;
}

// 改变形状（浅拷贝数据指针）
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    long long new_size = 1;
    for(int s : new_shape) new_size *= s;
    if (new_size != size_) throw std::runtime_error("Reshape size mismatch");

    Tensor res;
    res.data_ = data_; // 共享指针
    res.shape_ = new_shape;
    res.device_ = device_;
    res.size_ = size_;
    return res;
}

Device Tensor::device() const { return device_; }
const std::vector<int>& Tensor::shape() const { return shape_; }
long long Tensor::size() const { return size_; }
float* Tensor::data() { return data_.get(); }
const float* Tensor::data() const { return data_.get(); }

// 转移到 CPU
Tensor Tensor::cpu() const {
    if (device_ == Device::CPU) return *this;
    Tensor res(shape_, Device::CPU);
    if (size_ > 0) checkCuda(cudaMemcpy(res.data_.get(), data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToHost));
    return res;
}

// 转移到 GPU
Tensor Tensor::gpu() const {
    if (device_ == Device::GPU) return *this;
    Tensor res(shape_, Device::GPU);
    if (size_ > 0) checkCuda(cudaMemcpy(res.data_.get(), data_.get(), size_ * sizeof(float), cudaMemcpyHostToDevice));
    return res;
}

std::vector<float> Tensor::to_vector() const {
    if (device_ != Device::CPU) throw std::runtime_error("to_vector only on CPU");
    std::vector<float> vec(size_);
    if (size_ > 0) memcpy(vec.data(), data_.get(), size_ * sizeof(float));
    return vec;
}

void Tensor::zeros() {
    if (device_ == Device::CPU) memset(data_.get(), 0, size_ * sizeof(float));
    else checkCuda(cudaMemset(data_.get(), 0, size_ * sizeof(float)));
}

void Tensor::fill(const std::vector<float>& values) {
    if (values.size() != size_) throw std::runtime_error("Fill size mismatch");
    if (device_ == Device::CPU) memcpy(data_.get(), values.data(), size_ * sizeof(float));
    else checkCuda(cudaMemcpy(data_.get(), values.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::print() const {
    std::cout << "Tensor " << (device_==Device::CPU?"CPU":"GPU") << " shape=[";
    for(auto s : shape_) std::cout << s << ",";
    std::cout << "] ref_count=" << data_.use_count() << "\n";
}

void Tensor::add(const Tensor& A, const Tensor& B, Tensor& C) {
    // 启动 Add Kernel
    add_kernel<<<(A.size_+255)/256, 256>>>(A.data(), B.data(), C.data(), A.size_);
}


// ---------------------------------------------------------
// LAYERS IMPLEMENTATION (层方法实现)
// ---------------------------------------------------------

Tensor ReLU::forward(const Tensor& input) {
    Tensor output(input.shape(), input.device());
    if (input.device() == Device::CPU) { for(long long i=0;i<input.size();++i) output.data()[i]=std::max(0.0f,input.data()[i]); }
    else { relu_forward_kernel<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data(), output.data(), input.size()); }
    return output;
}

Tensor ReLU::backward(const Tensor& input, const Tensor& grad_output) {
    Tensor grad_input(input.shape(), input.device());
    if (input.device() == Device::CPU) { for(long long i=0;i<input.size();++i) grad_input.data()[i]=input.data()[i]>0?grad_output.data()[i]:0.0f; }
    else { relu_backward_kernel<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data(), grad_output.data(), grad_input.data(), input.size()); }
    return grad_input;
}

Tensor Sigmoid::forward(const Tensor& input) {
    Tensor output(input.shape(), input.device());
    if (input.device()==Device::CPU) { for(long long i=0;i<input.size();++i) output.data()[i]=1.0f/(1.0f+expf(-input.data()[i])); }
    else { sigmoid_forward_kernel<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data(), output.data(), input.size()); }
    return output;
}

Tensor Sigmoid::backward(const Tensor& input, const Tensor& grad_output) {
    Tensor grad_input(input.shape(), input.device());
    if(input.device()==Device::CPU){ for(long long i=0;i<input.size();++i){ float y=1.0f/(1.0f+expf(-input.data()[i])); grad_input.data()[i]=grad_output.data()[i]*y*(1.0f-y); }}
    else { sigmoid_backward_kernel<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data(), grad_output.data(), grad_input.data(), input.size()); }
    return grad_input;
}

// FC Forward
Tensor FullyConnected::forward(const Tensor& X, const Tensor& W, const Tensor& b) {
    if (X.device() == Device::CPU) throw std::runtime_error("Use GPU");
    init_cuda_resources();
    int N = X.shape()[0], Cin = X.shape()[1], Cout = W.shape()[0]; // 注意 W 形状可能是 [Cout, Cin]
    Tensor Y({N, Cout}, Device::GPU);
    
    // 转置 X (N, Cin) -> (Cin, N)，为了适应 cuBLAS 的列主序特性
    Tensor Xt({Cin, N}, Device::GPU);
    transpose_kernel<<<CudaGetBlocks(N*Cin), kCudaThreadsNum>>>(X.data(), Xt.data(), N, Cin);
    
    Tensor Cbuf({Cout, N}, Device::GPU);
    float alpha=1.0f, beta=0.0f;
    // 使用 cuBLAS 进行矩阵乘法: Cbuf = W * Xt
    // 逻辑上：Y = X * W^T
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, Cout, Cin, &alpha, Xt.data(), N, W.data(), Cin, &beta, Cbuf.data(), N);
    
    // 转置回 (N, Cout)
    transpose_kernel<<<CudaGetBlocks(N*Cout), kCudaThreadsNum>>>(Cbuf.data(), Y.data(), Cout, N);

    // 加上偏置
    bias_add_kernel<<<CudaGetBlocks(N*Cout), kCudaThreadsNum>>>(Y.data(), b.data(), N, Cout, 1);
    return Y;
}

// FC Backward
void FullyConnected::backward(const Tensor& X, const Tensor& W, const Tensor& dY, Tensor& dX, Tensor& dW, Tensor& db) {
    if (X.device() == Device::CPU) throw std::runtime_error("Use GPU");
    int N = X.shape()[0], Cin = X.shape()[1], Cout = W.shape()[0];
    init_cuda_resources(); float alpha=1.0f, beta=0.0f;
    
    // 1. 计算输入梯度 dX = dY * W
    dX = Tensor({N, Cin}, Device::GPU);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, Cin, N, Cout, &alpha, W.data(), Cin, dY.data(), Cout, &beta, dX.data(), Cin);
    
    // 2. 计算权重梯度 dW = dY^T * X
    Tensor dYT({Cout, N}, Device::GPU);
    transpose_kernel<<<CudaGetBlocks(N*Cout), kCudaThreadsNum>>>(dY.data(), dYT.data(), N, Cout);
    dW = Tensor({Cout, Cin}, Device::GPU);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, Cin, Cout, N, &alpha, X.data(), Cin, dYT.data(), N, &beta, dW.data(), Cin);

    // 3. 计算偏置梯度 db
    db = Tensor({Cout}, Device::GPU); db.zeros();
    fc_bias_backward_kernel<<<CudaGetBlocks(N*Cout), kCudaThreadsNum>>>(dY.data(), db.data(), N, Cout);
}

IntBuffer::IntBuffer() : size(0) {}
IntBuffer::IntBuffer(size_t n) { alloc_on_gpu(n); }
void IntBuffer::alloc_on_gpu(size_t n) {
    size = n; int* p; checkCuda(cudaMalloc(&p, n * sizeof(int)), "IntBuffer Malloc");
    ptr_ = std::shared_ptr<int>(p, [](int* ptr){ if(ptr) cudaFree(ptr); });
}
int* IntBuffer::get() const { return ptr_.get(); }

Tensor MaxPool2x2::forward(const Tensor& X, IntBuffer &mask_out) {
    int N = X.shape()[0], C = X.shape()[1], H = X.shape()[2], W = X.shape()[3];
    int out_h = H/2, out_w = W/2;
    Tensor Y({N, C, out_h, out_w}, Device::GPU);
    mask_out.alloc_on_gpu(Y.size());
    maxpool2x2_forward_kernel<<<CudaGetBlocks(Y.size()), kCudaThreadsNum>>>(X.data(), N, C, H, W, out_h, out_w, Y.data(), mask_out.get());
    return Y;
}

Tensor MaxPool2x2::backward(const Tensor& X, const Tensor& dY, const IntBuffer &mask_buf) {
    int N = X.shape()[0], C = X.shape()[1], H = X.shape()[2], W = X.shape()[3];
    int out_h = H/2, out_w = W/2;
    Tensor dX({N, C, H, W}, Device::GPU); dX.zeros();
    maxpool2x2_backward_kernel<<<CudaGetBlocks(dY.size()), kCudaThreadsNum>>>(dY.data(), mask_buf.get(), N, C, H, W, out_h, out_w, dX.data());
    return dX;
}

std::pair<float, Tensor> SoftmaxCrossEntropy::forward(const Tensor& logits, const std::vector<int>& labels) {
    int N = logits.shape()[0], C = logits.shape()[1];
    Tensor probs({N, C}, Device::GPU);

    // 拷贝标签到 GPU
    int *d_labels; cudaMalloc(&d_labels, N * sizeof(int));
    cudaMemcpy(d_labels, labels.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // 分配 Loss 累加器
    float *d_loss, h_loss = 0; cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    softmax_ce_kernel<<<(N+255)/256, 256>>>(logits.data(), probs.data(), d_labels, d_loss, N, C);

    // 拷贝 Loss 回 CPU
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_labels); cudaFree(d_loss);
    return { h_loss / N, probs }; // 返回平均 Loss
}

Tensor SoftmaxCrossEntropy::backward(const Tensor& probs, const std::vector<int>& labels) {
    int N = probs.shape()[0], C = probs.shape()[1];
    Tensor grad({N, C}, Device::GPU);
    int *d_labels; cudaMalloc(&d_labels, N * sizeof(int));
    cudaMemcpy(d_labels, labels.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    // 计算梯度
    ce_backward_kernel<<<(N*C+255)/256, 256>>>(grad.data(), probs.data(), d_labels, N, C);
    cudaFree(d_labels);
    return grad;
}

Tensor Conv2d::forward(const Tensor& X, const Tensor& W, const Tensor& b, int stride, int pad) {
    // 1. 计算输出尺寸
    int N = X.shape()[0], Cin = X.shape()[1], H = X.shape()[2], Wd = X.shape()[3];
    int Cout = W.shape()[0], kh = W.shape()[2], kw = W.shape()[3];
    int H_out = (H + 2*pad - kh) / stride + 1;
    int W_out = (Wd + 2*pad - kw) / stride + 1;

    int K = Cin * kh * kw, col_cols = H_out * W_out;
    init_cuda_resources();

    // 2. Im2Col: 将每个卷积窗口展平成列
    Tensor col_dev({N*K, col_cols}, Device::GPU);
    im2col_kernel_batch<<<CudaGetBlocks(N*K*col_cols), kCudaThreadsNum>>>(X.data(), N, Cin, H, Wd, kh, kw, pad, pad, stride, stride, H_out, W_out, col_dev.data());

    // 3. GEMM: 使用矩阵乘法完成卷积运算
    Tensor Y({N, Cout, H_out, W_out}, Device::GPU);
    Tensor outbuf({Cout, col_cols}, Device::GPU);
    float alpha=1.0f, beta=0.0f;
    for(int n=0; n<N; ++n) {
        float* col_n = col_dev.data() + n * K * col_cols; // 当前样本的 im2col 矩阵
        // out = W * col
        cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, col_cols, Cout, K, &alpha, col_n, col_cols, W.data(), K, &beta, outbuf.data(), col_cols);
        // 拷贝结果到 Y 的对应位置
        checkCuda(cudaMemcpy(Y.data() + n*Cout*col_cols, outbuf.data(), Cout*col_cols*sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // 4. Bias Add
    bias_add_kernel<<<CudaGetBlocks(N*Cout*col_cols), kCudaThreadsNum>>>(Y.data(), b.data(), N, Cout, col_cols);
    return Y;
}

void Conv2d::backward(const Tensor& X, const Tensor& W, const Tensor& dY, Tensor& dX, Tensor& dW, Tensor& db, int stride, int pad) {
    int N = X.shape()[0], Cin = X.shape()[1], H = X.shape()[2], Wd = X.shape()[3];
    int Cout = W.shape()[0], kh = W.shape()[2], kw = W.shape()[3];
    int H_out = dY.shape()[2], W_out = dY.shape()[3];
    int K = Cin * kh * kw, col_cols = H_out * W_out;

    init_cuda_resources();
    // 重新 Im2Col 输入 X，用于计算 dW
    Tensor col_dev({N * K, col_cols}, Device::GPU);
    im2col_kernel_batch<<<CudaGetBlocks((long)N * K * col_cols), kCudaThreadsNum>>>(X.data(), N, Cin, H, Wd, kh, kw, pad, pad, stride, stride, H_out, W_out, col_dev.data());

    // 1. 计算 dW
    dW = Tensor({Cout, Cin, kh, kw}, Device::GPU); dW.zeros();
    Tensor dW_acc({Cout, K}, Device::GPU);
    Tensor colnT({col_cols, K}, Device::GPU);
    float alpha = 1.0f, beta = 0.0f;
    for (int n=0; n<N; ++n) {
        const float* col_n = col_dev.data() + n * K * col_cols;
        const float* dY_n = dY.data() + n * Cout * col_cols;
        transpose_kernel<<<CudaGetBlocks(K*col_cols), kCudaThreadsNum>>>(col_n, colnT.data(), K, col_cols);
        cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, K, Cout, col_cols, &alpha, colnT.data(), K, dY_n, col_cols, &beta, dW_acc.data(), K);
        cublasSaxpy(g_cublas, Cout*K, &alpha, dW_acc.data(), 1, dW.data(), 1);
    }

    // 2. 计算 db
    db = Tensor({Cout}, Device::GPU); db.zeros();
    conv2d_bias_backward_kernel<<<CudaGetBlocks(N*Cout*col_cols), kCudaThreadsNum>>>(dY.data(), db.data(), N, Cout, col_cols);

    // 3. 计算 dX (使用 Col2Im)
    dX = Tensor({N, Cin, H, Wd}, Device::GPU); dX.zeros();
    Tensor Wmat({K, Cout}, Device::GPU);
    transpose_kernel<<<CudaGetBlocks(Cout*K), kCudaThreadsNum>>>(W.data(), Wmat.data(), Cout, K);
    Tensor grad_col_dev({K, col_cols}, Device::GPU);
    for (int n=0; n<N; ++n) {
        const float* dY_n = dY.data() + n * Cout * col_cols;
        // 计算梯度对应的 col 形式
        cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, col_cols, K, Cout, &alpha, dY_n, col_cols, Wmat.data(), Cout, &beta, grad_col_dev.data(), col_cols);
        float* dX_ptr = dX.data() + n * Cin * H * Wd;
        // 将 col 形式还原回图片结构
        col2im_kernel_batch_atomic<<<CudaGetBlocks(K*col_cols), kCudaThreadsNum>>>(grad_col_dev.data(), 1, Cin, H, Wd, kh, kw, pad, pad, stride, stride, H_out, W_out, dX_ptr);
    }
}

// ---------------------------------------------------------
// OPTIMIZER
// ---------------------------------------------------------

void Optimizer::sgd_momentum(Tensor& param, const Tensor& grad, Tensor& velocity, float lr, float momentum) {
    if (param.device() != Device::GPU || grad.device() != Device::GPU || velocity.device() != Device::GPU) {
        throw std::runtime_error("SGD optimizer requires all tensors on GPU");
    }
    sgd_momentum_kernel<<<CudaGetBlocks(param.size()), kCudaThreadsNum>>>(
        param.data(), grad.data(), velocity.data(), lr, momentum, param.size()
    );
}