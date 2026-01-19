#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   
#include <pybind11/numpy.h> 
#include "tensor_lib.h"

namespace py = pybind11;

// 定义 Python 模块名为 mytensor
PYBIND11_MODULE(mytensor, m) {
    // 导出 Device 枚举
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    // 导出 IntBuffer (主要用于内部传递 mask，Python 端只需持有对象)
    py::class_<IntBuffer>(m, "IntBuffer")
        .def(py::init<size_t>());

    // 导出 Tensor 类
    py::class_<Tensor>(m, "Tensor")
        // 构造函数绑定
        .def(py::init<const std::vector<int>&, Device>(), py::arg("shape"), py::arg("device")=Device::CPU)
        .def(py::init<const std::vector<int>&, const std::vector<float>&, Device>(), 
            py::arg("shape"), py::arg("data"), py::arg("device")=Device::CPU)
        // 成员函数绑定
        .def("shape", &Tensor::shape)
        .def("device", &Tensor::device)
        .def("size", &Tensor::size)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("clone", &Tensor::clone)
        .def("reshape", &Tensor::reshape)
        .def("to_vector", &Tensor::to_vector)
        .def("fill", &Tensor::fill)
        .def("zeros", &Tensor::zeros)
        .def("print", &Tensor::print)
        .def_static("add", &Tensor::add);

    // 导出 ReLU
    py::class_<ReLU>(m, "ReLU")
        .def_static("forward", &ReLU::forward)
        .def_static("backward", &ReLU::backward);

    // 导出 Sigmoid
    py::class_<Sigmoid>(m, "Sigmoid")
        .def_static("forward", &Sigmoid::forward)
        .def_static("backward", &Sigmoid::backward);

    // 导出 FullyConnected
    py::class_<FullyConnected>(m, "FullyConnected")
        .def_static("forward", &FullyConnected::forward)
        // Backward 在 C++ 中是通过引用参数返回结果 (Tensor& dX...)
        // 在 Python 中习惯返回 Tuple，所以这里用 Lambda 封装一下
        .def_static("backward", [](const Tensor& X, const Tensor& W, const Tensor& dY) {
            Tensor dX, dW, db;
            FullyConnected::backward(X, W, dY, dX, dW, db);
            return std::make_tuple(dX, dW, db); // 返回 (dX, dW, db)
        });

    // 导出 Conv2d
    py::class_<Conv2d>(m, "Conv2d")
        .def_static("forward", &Conv2d::forward, py::arg("X"), py::arg("W"), py::arg("b"), py::arg("stride")=1, py::arg("pad")=0)
        // 同样使用 Lambda 封装 Backward，返回 Tuple
        .def_static("backward", [](const Tensor& X, const Tensor& W, const Tensor& dY, int stride, int pad) {
            Tensor dX, dW, db;
            Conv2d::backward(X, W, dY, dX, dW, db, stride, pad);
            return std::make_tuple(dX, dW, db);
        }, py::arg("X"), py::arg("W"), py::arg("dY"), py::arg("stride")=1, py::arg("pad")=0);

    // 导出 MaxPool2x2
    py::class_<MaxPool2x2>(m, "MaxPool2x2")
        // Forward 需要返回结果 Y 和 mask (用于反向传播)
        .def_static("forward", [](const Tensor& X) {
            IntBuffer mask(0);
            Tensor Y = MaxPool2x2::forward(X, mask);
            return std::make_pair(Y, mask);
        })
        .def_static("backward", &MaxPool2x2::backward);

    // 导出 SoftmaxCrossEntropy
    py::class_<SoftmaxCrossEntropy>(m, "SoftmaxCrossEntropy")
        .def_static("forward", &SoftmaxCrossEntropy::forward)
        .def_static("backward", &SoftmaxCrossEntropy::backward);
        
    // 导出 Optimizer
    py::class_<Optimizer>(m, "Optimizer")
        .def_static("sgd_momentum", &Optimizer::sgd_momentum);
}