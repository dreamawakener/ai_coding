%%writefile src/tensor_lib.cu
#include "tensor_lib.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#define kCudaThreadsNum 256

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

inline void checkCuda(cudaError_t e, const char* msg = "") {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(e) << std::endl;
        throw std::runtime_error(cudaGetErrorString(e));
    }
}

static cublasHandle_t g_cublas = nullptr;
static bool is_cuda_initialized = false;

static void init_cuda_resources() {
    if (!is_cuda_initialized) {
        cublasCreate(&g_cublas);
        is_cuda_initialized = true;
    }
}

// ---------------------------------------------------------
// KERNELS
// ---------------------------------------------------------

__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx] > 0 ? input[idx] : 0.0f;
}

__global__ void relu_backward_kernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
}

__global__ void sigmoid_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

__global__ void sigmoid_backward_kernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = 1.0f / (1.0f + expf(-input[idx]));
        grad_input[idx] = grad_output[idx] * y * (1.0f - y);
    }
}

__global__ void bias_add_kernel(float* out, const float* bias, int N, int Cout, int col_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * col_cols;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int tmp = i / col_cols;
        int co = tmp % Cout;
        out[i] += bias[co];
    }
}

__global__ void softmax_forward_kernel(const float* __restrict__ logits, float* __restrict__ probs, int N, int C) {
    int n = blockIdx.x;
    if (n >= N) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local_max = -1e30f;
    for (int i = tid; i < C; i += blockDim.x) {
        float v = logits[n * (long)C + i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float mx = sdata[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float e = expf(logits[n * (long)C + i] - mx);
        probs[n * (long)C + i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();
    for (int i = tid; i < C; i += blockDim.x) {
        probs[n * (long)C + i] /= sum;
    }
}

__global__ void cross_entropy_loss_kernel(const float* probs, const int* labels, float* out_losses, int N, int C) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lb = labels[n];
    float p = probs[n * (long)C + lb];
    out_losses[n] = -logf(fmaxf(p, 1e-12f));
}

__global__ void cross_entropy_backward_kernel(const float* probs, const int* labels, float* grad_logits, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int n = i / C;
        int c = i % C;
        float g = probs[i];
        if (labels[n] == c) g -= 1.0f;
        grad_logits[i] = g / float(N);
    }
}

__global__ void transpose_kernel(const float* src, float* dst, int R, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = R * C;
    if (idx >= total) return;
    int r = idx / C;
    int c = idx % C;
    dst[c * (size_t)R + r] = src[r * (size_t)C + c];
}

__global__ void im2col_kernel_batch(const float* data_im, int N, int C, int H, int W,
                                    int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w,
                                    int H_out, int W_out, float* data_col) {
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
        float val = 0.0f;
        if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
            long src_idx = n * (long)C * H * W + c * (long)H * W + im_row * W + im_col;
            val = data_im[src_idx];
        }
        long dst_idx = n * (long)K * (H_out * W_out) + k * (long)(H_out * W_out) + out_pos;
        data_col[dst_idx] = val;
    }
}

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
            atomicAdd(&data_im_grad[dst_idx], data_col[src_idx]);
        }
    }
}

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
        mask[idx] = best_idx;
    }
}

__global__ void maxpool2x2_backward_kernel(const float* grad_out, const int* mask, int N, int C, int H, int W,
                                            int out_h, int out_w, float* grad_in) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    for (int idx = index; idx < total; idx += blockDim.x * gridDim.x) {
        int in_pos = mask[idx];
        if (in_pos >= 0) atomicAdd(&grad_in[in_pos], grad_out[idx]);
    }
}

__global__ void sgd_momentum_kernel(float* param, const float* grad, float* velocity, float lr, float momentum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        float v = velocity[i];
        float p = param[i];
        
        v = momentum * v + g;
        p = p - lr * v;
        
        velocity[i] = v;
        param[i] = p;
    }
}

// ---------------------------------------------------------
// TENSOR IMPLEMENTATION (SHALLOW COPY)
// ---------------------------------------------------------

Tensor::Tensor() : data_(nullptr), shape_(), device_(Device::CPU), size_(0) {}

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
        data_ = std::shared_ptr<float>(ptr, [](float* p){ if(p) cudaFree(p); });
    }
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data, Device device) : shape_(shape), device_(device) {
    size_ = 1; for (int n : shape) size_ *= n;
    
    if (device == Device::CPU) {
        data_ = std::shared_ptr<float>(new float[size_], std::default_delete<float[]>());
        memcpy(data_.get(), data.data(), size_ * sizeof(float));
    } else {
        init_cuda_resources(); 
        float* ptr;
        checkCuda(cudaMalloc(&ptr, size_ * sizeof(float)));
        checkCuda(cudaMemcpy(ptr, data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
        data_ = std::shared_ptr<float>(ptr, [](float* p){ if(p) cudaFree(p); });
    }
}

Tensor Tensor::clone() const {
    Tensor res; 
    res.shape_ = shape_; 
    res.device_ = device_; 
    res.size_ = size_;
    if (size_ == 0) return res;

    if (device_ == Device::CPU) {
        res.data_ = std::shared_ptr<float>(new float[size_], std::default_delete<float[]>());
        memcpy(res.data_.get(), data_.get(), size_ * sizeof(float));
    } else {
        init_cuda_resources(); 
        float* ptr;
        checkCuda(cudaMalloc(&ptr, size_ * sizeof(float)));
        checkCuda(cudaMemcpy(ptr, data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToDevice));
        res.data_ = std::shared_ptr<float>(ptr, [](float* p){ if(p) cudaFree(p); });
    }
    return res;
}

Device Tensor::device() const { return device_; }
const std::vector<int>& Tensor::shape() const { return shape_; }
long long Tensor::size() const { return size_; }
float* Tensor::data() { return data_.get(); }
const float* Tensor::data() const { return data_.get(); }

Tensor Tensor::cpu() const {
    if (device_ == Device::CPU) return *this;
    Tensor res(shape_, Device::CPU);
    if (size_ > 0) checkCuda(cudaMemcpy(res.data_.get(), data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToHost));
    return res;
}

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

// ---------------------------------------------------------
// LAYERS IMPLEMENTATION
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

Tensor FullyConnected::forward(const Tensor& X, const Tensor& W, const Tensor& b) {
    if (X.device() == Device::CPU) throw std::runtime_error("Use GPU");
    init_cuda_resources();
    int N = X.shape()[0], Cin = X.shape()[1], Cout = W.shape()[0];
    Tensor Y({N, Cout}, Device::GPU);
    Tensor Xt({Cin, N}, Device::GPU);
    transpose_kernel<<<CudaGetBlocks(N*Cin), kCudaThreadsNum>>>(X.data(), Xt.data(), N, Cin);
    Tensor Cbuf({Cout, N}, Device::GPU);
    float alpha=1.0f, beta=0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, Cout, Cin, &alpha, Xt.data(), N, W.data(), Cin, &beta, Cbuf.data(), N);
    transpose_kernel<<<CudaGetBlocks(N*Cout), kCudaThreadsNum>>>(Cbuf.data(), Y.data(), Cout, N);
    Tensor bias_dev({Cout}, Device::GPU);
    bias_dev.fill(b.device()==Device::CPU ? b.to_vector() : b.cpu().to_vector());
    bias_add_kernel<<<CudaGetBlocks(N*Cout), kCudaThreadsNum>>>(Y.data(), bias_dev.data(), N, Cout, 1);
    return Y;
}

void FullyConnected::backward(const Tensor& X, const Tensor& W, const Tensor& dY, Tensor& dX, Tensor& dW, Tensor& db) {
    if (X.device() == Device::CPU) throw std::runtime_error("Use GPU");
    int N = X.shape()[0], Cin = X.shape()[1], Cout = W.shape()[0];
    init_cuda_resources(); float alpha=1.0f, beta=0.0f;
    dX = Tensor({N, Cin}, Device::GPU);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, Cin, N, Cout, &alpha, W.data(), Cin, dY.data(), Cout, &beta, dX.data(), Cin);
    Tensor dYT({Cout, N}, Device::GPU);
    transpose_kernel<<<CudaGetBlocks(N*Cout), kCudaThreadsNum>>>(dY.data(), dYT.data(), N, Cout);
    dW = Tensor({Cout, Cin}, Device::GPU);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, Cin, Cout, N, &alpha, X.data(), Cin, dYT.data(), N, &beta, dW.data(), Cin);
    
    // --------------------------------------------------------
    // [CRITICAL FIX] Ensure db is returned as GPU Tensor
    // --------------------------------------------------------
    Tensor dY_cpu = dY.cpu();
    std::vector<float> dbvec(Cout, 0.0f);
    for (int n=0; n<N; ++n) for(int c=0; c<Cout; ++c) dbvec[c] += dY_cpu.data()[n * Cout + c];
    
    // Create on GPU immediately
    db = Tensor({Cout}, Device::GPU);
    // fill handles host->device copy automatically
    db.fill(dbvec); 
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

Tensor SoftMax::forward(const Tensor& X) {
    int N = X.shape()[0], C = X.shape()[1];
    Tensor probs({N, C}, Device::GPU);
    softmax_forward_kernel<<<N, kCudaThreadsNum, kCudaThreadsNum*sizeof(float)>>>(X.data(), probs.data(), N, C);
    return probs;
}

float SoftMax::cross_entropy_with_softmax(const Tensor& logits, const std::vector<int>& labels, Tensor& grad_logits) {
    int N = logits.shape()[0], C = logits.shape()[1];
    Tensor probs = forward(logits);
    int* dlabels; cudaMalloc(&dlabels, N*sizeof(int));
    cudaMemcpy(dlabels, labels.data(), N*sizeof(int), cudaMemcpyHostToDevice);
    Tensor losses({N}, Device::GPU);
    cross_entropy_loss_kernel<<<CudaGetBlocks(N), kCudaThreadsNum>>>(probs.data(), dlabels, losses.data(), N, C);
    Tensor cpu_loss = losses.cpu();
    float total_loss = 0; for(long long i=0; i<cpu_loss.size(); ++i) total_loss += cpu_loss.data()[i];
    grad_logits = Tensor({N, C}, Device::GPU);
    cross_entropy_backward_kernel<<<CudaGetBlocks(N*C), kCudaThreadsNum>>>(probs.data(), dlabels, grad_logits.data(), N, C);
    cudaFree(dlabels);
    return total_loss / N;
}

Tensor Conv2d::forward(const Tensor& X, const Tensor& W, const Tensor& b) {
    int N = X.shape()[0], Cin = X.shape()[1], H = X.shape()[2], Wd = X.shape()[3];
    int Cout = W.shape()[0], kh=3, kw=3, H_out = H, W_out = Wd, K = Cin * kh * kw, col_cols = H_out * W_out;
    init_cuda_resources();
    Tensor col_dev({N*K, col_cols}, Device::GPU);
    im2col_kernel_batch<<<CudaGetBlocks(N*K*col_cols), kCudaThreadsNum>>>(X.data(), N, Cin, H, Wd, kh, kw, 1, 1, 1, 1, H_out, W_out, col_dev.data());
    Tensor Y({N, Cout, H_out, W_out}, Device::GPU);
    Tensor outbuf({Cout, col_cols}, Device::GPU);
    float alpha=1.0f, beta=0.0f;
    for(int n=0; n<N; ++n) {
        float* col_n = col_dev.data() + n * K * col_cols;
        cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, col_cols, Cout, K, &alpha, col_n, col_cols, W.data(), K, &beta, outbuf.data(), col_cols);
        checkCuda(cudaMemcpy(Y.data() + n*Cout*col_cols, outbuf.data(), Cout*col_cols*sizeof(float), cudaMemcpyDeviceToDevice));
    }
    Tensor bias_dev({Cout}, Device::GPU);
    bias_dev.fill(b.device()==Device::CPU ? b.to_vector() : b.cpu().to_vector());
    bias_add_kernel<<<CudaGetBlocks(N*Cout*col_cols), kCudaThreadsNum>>>(Y.data(), bias_dev.data(), N, Cout, col_cols);
    return Y;
}

void Conv2d::backward(const Tensor& X, const Tensor& W, const Tensor& dY, Tensor& dX, Tensor& dW, Tensor& db) {
    int N = X.shape()[0], Cin = X.shape()[1], H = X.shape()[2], Wd = X.shape()[3];
    int Cout = W.shape()[0], kh=3, kw=3, pad=1, stride=1, H_out = H, W_out = Wd, K = Cin * kh * kw, col_cols = H_out * W_out;
    init_cuda_resources();
    Tensor col_dev({N * K, col_cols}, Device::GPU);
    im2col_kernel_batch<<<CudaGetBlocks((long)N * K * col_cols), kCudaThreadsNum>>>(X.data(), N, Cin, H, Wd, kh, kw, pad, pad, stride, stride, H_out, W_out, col_dev.data());
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
    
    // --------------------------------------------------------
    // [CRITICAL FIX] Ensure db is returned as GPU Tensor
    // --------------------------------------------------------
    Tensor dY_cpu = dY.cpu();
    std::vector<float> dbvec(Cout, 0.0f);
    for (int n=0; n<N; ++n) for (int co=0; co<Cout; ++co) for (int idx=0; idx<col_cols; ++idx) dbvec[co] += dY_cpu.data()[n * Cout * col_cols + co * col_cols + idx];
    
    // Create on GPU immediately
    db = Tensor({Cout}, Device::GPU);
    // fill handles host->device copy automatically
    db.fill(dbvec);

    dX = Tensor({N, Cin, H, Wd}, Device::GPU); dX.zeros();
    Tensor Wmat({K, Cout}, Device::GPU);
    transpose_kernel<<<CudaGetBlocks(Cout*K), kCudaThreadsNum>>>(W.data(), Wmat.data(), Cout, K);
    Tensor grad_col_dev({K, col_cols}, Device::GPU);
    for (int n=0; n<N; ++n) {
        const float* dY_n = dY.data() + n * Cout * col_cols;
        cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, col_cols, K, Cout, &alpha, dY_n, col_cols, Wmat.data(), Cout, &beta, grad_col_dev.data(), col_cols);
        float* dX_ptr = dX.data() + n * Cin * H * Wd;
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