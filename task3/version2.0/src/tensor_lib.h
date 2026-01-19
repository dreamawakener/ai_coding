#pragma once
#include <memory>
#include <vector>
#include <string>
#include <iostream>

// 定义设备类型：CPU 或 GPU
enum class Device {
    CPU,
    GPU
};

class Tensor {
public:
    // 使用 shared_ptr 自动管理内存，避免手动 new/delete 造成的内存泄漏
    // 在 GPU 模式下，这个指针指向显存地址；CPU 模式下指向主机内存
    std::shared_ptr<float> data_;
    std::vector<int> shape_; // 张量的形状，例如 [Batch, Channels, Height, Width]
    Device device_;          // 当前张量所在的设备
    long long size_;         // 元素总数

    // 构造函数
    Tensor();
    Tensor(const std::vector<int>& shape, Device device = Device::CPU);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data, Device device = Device::CPU);
    Tensor(const Tensor&) = default; // 默认拷贝构造（注意：shared_ptr 拷贝是浅拷贝，引用计数+1）

    Tensor clone() const; // 深拷贝：分配新内存并复制数据
    Tensor reshape(const std::vector<int>& new_shape) const; // 改变形状（数据共享）

    // 基础属性访问
    Device device() const;
    const std::vector<int>& shape() const;
    long long size() const;
    float* data();
    const float* data() const;

    // 设备间数据传输
    Tensor cpu() const; // 将数据从 GPU 复制到 CPU
    Tensor gpu() const; // 将数据从 CPU 复制到 GPU

    std::vector<float> to_vector() const;           // 转为 C++ vector 用于调试或输出
    void zeros();                                   // 全零初始化
    void fill(const std::vector<float>& values);    // 用给定数据填充
    void print() const;                             // 打印元数据
    
    // 静态加法函数：C = A + B
    static void add(const Tensor& A, const Tensor& B, Tensor& C);
};

// --- 激活函数 ---
class ReLU {
public:
    static Tensor forward(const Tensor& input);
    // Backward 需要输入原始输入数据（用于判断 > 0）和上游传来的梯度
    static Tensor backward(const Tensor& input, const Tensor& grad_output);
};

class Sigmoid {
public:
    static Tensor forward(const Tensor& input);
    static Tensor backward(const Tensor& input, const Tensor& grad_output);
};

// --- 全连接层 ---
class FullyConnected {
public:
    // Y = X * W + b
    static Tensor forward(const Tensor& X, const Tensor& W, const Tensor& b);
    // 反向传播：同时计算对输入 X、权重 W、偏置 b 的梯度
    // dX, dW, db 是输出参数
    static void backward(   const Tensor& X, const Tensor& W, const Tensor& dY,
                            Tensor& dX, Tensor& dW, Tensor& db);
};

// --- 卷积层 ---
class Conv2d {
public:
    // Im2Col + GEMM 实现的卷积
    static Tensor forward(const Tensor& X, const Tensor& W, const Tensor& b, int stride=1, int pad=0);
    static void backward(   const Tensor& X, const Tensor& W, const Tensor& dY,
                            Tensor& dX, Tensor& dW, Tensor& db, int stride=1, int pad=0);
};

// --- 辅助结构：用于 MaxPool 记录最大值索引 ---
struct IntBuffer {
    std::shared_ptr<int> ptr_;
    size_t size;
    IntBuffer();
    IntBuffer(size_t n);
    void alloc_on_gpu(size_t n);
    int* get() const;
};

// --- 池化层 ---
class MaxPool2x2 {
public:
    // forward 需要输出一个 mask (IntBuffer)，记录最大值位置，供 backward 使用
    static Tensor forward(const Tensor& X, IntBuffer &mask_out);
    static Tensor backward(const Tensor& X, const Tensor& dY, const IntBuffer &mask_buf);
};

// --- 损失函数 ---
// Softmax 和 CrossEntropy 结合在一起计算，数值上更稳定
class SoftmaxCrossEntropy {
public:
    // 返回 pair: (平均 Loss 值, Softmax后的概率分布 Tensor)
    static std::pair<float, Tensor> forward(const Tensor& logits, const std::vector<int>& labels);
    static Tensor backward(const Tensor& probs, const std::vector<int>& labels);
};

// --- 优化器 ---
class Optimizer {
public:
    // SGD + Momentum 更新参数
    // velocity 是动量缓存，需要由外部维护状态
    static void sgd_momentum(Tensor& param, const Tensor& grad, Tensor& velocity, float lr, float momentum);
};