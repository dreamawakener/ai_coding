#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // 必须包含此头文件以支持 std::vector 转换
#include <pybind11/numpy.h> // 支持 numpy 数组转换
#include "tensor_lib.h"

namespace py = pybind11;

PYBIND11_MODULE(mytensor, m) {
    // 暴露 Device 枚举
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    // 暴露 IntBuffer
    py::class_<IntBuffer>(m, "IntBuffer")
        .def(py::init<size_t>());

    // 暴露 Tensor 类及其方法
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, Device>(), py::arg("shape"), py::arg("device")=Device::CPU)
        .def(py::init<const std::vector<int>&, const std::vector<float>&, Device>(), 
             py::arg("shape"), py::arg("data"), py::arg("device")=Device::CPU)
        .def(py::init([](py::array_t<float> array, Device device) {
            py::buffer_info buf = array.request();
            std::vector<int> shape;
            for (auto s : buf.shape) shape.push_back(static_cast<int>(s));
            std::vector<float> data(static_cast<float*>(buf.ptr), static_cast<float*>(buf.ptr) + buf.size);
            return new Tensor(shape, data, device);
        }), py::arg("array"), py::arg("device")=Device::CPU)
        .def("shape", &Tensor::shape)
        .def("device", &Tensor::device)
        .def("size", &Tensor::size)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("clone", &Tensor::clone)
        .def("reshape", &Tensor::reshape)
        .def("to_vector", &Tensor::to_vector)
        .def("fill", &Tensor::fill)
        .def("zeros", &Tensor::zeros)    // 修复你当前的报错
        .def("numpy", [](const Tensor& t) {
            std::vector<float> vec = t.cpu().to_vector();
            std::vector<ssize_t> shape;
            for(auto s : t.shape()) shape.push_back(s);
            return py::array_t<float>(shape, vec.data());
        })
        .def("print", &Tensor::print);

    // 暴露 ReLU
    py::class_<ReLU>(m, "ReLU")
        .def_static("forward", &ReLU::forward)
        .def_static("backward", &ReLU::backward);

    // 暴露 Sigmoid
    py::class_<Sigmoid>(m, "Sigmoid")
        .def_static("forward", &Sigmoid::forward)
        .def_static("backward", &Sigmoid::backward);

    // 暴露 FullyConnected
    py::class_<FullyConnected>(m, "FullyConnected")
        .def_static("forward", &FullyConnected::forward)
        .def_static("backward", [](const Tensor& X, const Tensor& W, const Tensor& dY) {
            Tensor dX, dW, db;
            FullyConnected::backward(X, W, dY, dX, dW, db);
            return std::make_tuple(dX, dW, db);
        });

    // 暴露 Conv2d
    py::class_<Conv2d>(m, "Conv2d")
        .def_static("forward", &Conv2d::forward, py::arg("X"), py::arg("W"), py::arg("b"), py::arg("stride")=1, py::arg("pad")=0)
        .def_static("backward", [](const Tensor& X, const Tensor& W, const Tensor& dY, int stride, int pad) {
            Tensor dX, dW, db;
            Conv2d::backward(X, W, dY, dX, dW, db, stride, pad);
            return std::make_tuple(dX, dW, db);
        }, py::arg("X"), py::arg("W"), py::arg("dY"), py::arg("stride")=1, py::arg("pad")=0);

    // 暴露 MaxPool2x2
    py::class_<MaxPool2x2>(m, "MaxPool2x2")
        .def_static("forward", [](const Tensor& X) {
            IntBuffer mask(0);
            Tensor Y = MaxPool2x2::forward(X, mask);
            return std::make_pair(Y, mask);
        })
        .def_static("backward", &MaxPool2x2::backward);

    // --- 核心修改：暴露 SoftMax 接口 ---
    py::class_<SoftMax>(m, "SoftMax")
        .def_static("forward", &SoftMax::forward) // 对应 static Tensor forward(const Tensor& X)
        .def_static("cross_entropy", [](const Tensor& logits, const std::vector<int>& labels) {
            Tensor grad;
            // 调用 C++ 中的 cross_entropy_with_softmax
            float loss = SoftMax::cross_entropy_with_softmax(logits, labels, grad);
            // 将 loss 和 生成的梯度 grad 以 tuple 形式返回给 Python
            return std::make_pair(loss, grad);
        });
        
    // 暴露 Optimizer
    py::class_<Optimizer>(m, "Optimizer")
        .def_static("sgd_momentum", &Optimizer::sgd_momentum);
}