%%writefile src/tensor_lib.h
#pragma once
#include <memory>
#include <vector>
#include <string>
#include <iostream>

enum class Device {
    CPU,
    GPU
};

class Tensor {
public:
    std::shared_ptr<float> data_;
    std::vector<int> shape_;
    Device device_;
    long long size_;

    Tensor();
    Tensor(const std::vector<int>& shape, Device device = Device::CPU);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data, Device device = Device::CPU);
    Tensor(const Tensor&) = default; // 浅拷贝

    Tensor clone() const; // 深拷贝

    Device device() const;
    const std::vector<int>& shape() const;
    long long size() const;
    float* data();
    const float* data() const;

    Tensor cpu() const;
    Tensor gpu() const;
    std::vector<float> to_vector() const;
    void zeros();
    void fill(const std::vector<float>& values);
    void print() const;
};

class ReLU {
public:
    static Tensor forward(const Tensor& input);
    static Tensor backward(const Tensor& input, const Tensor& grad_output);
};

class Sigmoid {
public:
    static Tensor forward(const Tensor& input);
    static Tensor backward(const Tensor& input, const Tensor& grad_output);
};

class FullyConnected {
public:
    static Tensor forward(const Tensor& X, const Tensor& W, const Tensor& b);
    static void backward(const Tensor& X, const Tensor& W, const Tensor& dY,
                         Tensor& dX, Tensor& dW, Tensor& db);
};

class Conv2d {
public:
    static Tensor forward(const Tensor& X, const Tensor& W, const Tensor& b);
    static void backward(const Tensor& X, const Tensor& W, const Tensor& dY,
                         Tensor& dX, Tensor& dW, Tensor& db);
};

struct IntBuffer {
    std::shared_ptr<int> ptr_;
    size_t size;
    IntBuffer();
    IntBuffer(size_t n);
    void alloc_on_gpu(size_t n);
    int* get() const;
};

class MaxPool2x2 {
public:
    static Tensor forward(const Tensor& X, IntBuffer &mask_out);
    static Tensor backward(const Tensor& X, const Tensor& dY, const IntBuffer &mask_buf);
};

class SoftMax {
public:
    static Tensor forward(const Tensor& X);
    static float cross_entropy_with_softmax(const Tensor& logits, const std::vector<int>& labels, Tensor& grad_logits);
};

class Optimizer {
public:
    static void sgd_momentum(Tensor& param, const Tensor& grad, Tensor& velocity, float lr, float momentum);
};