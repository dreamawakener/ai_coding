#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor_lib.h"

namespace py = pybind11;

PYBIND11_MODULE(mytensor, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<IntBuffer>(m, "IntBuffer")
        .def(py::init<size_t>());

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
        .def("to_vector", &Tensor::to_vector)
        .def("fill", &Tensor::fill)
        .def("to", [](const Tensor& t, Device device) {
            if (device == Device::CPU) return t.cpu();
            else if (device == Device::GPU) return t.gpu();
            else throw std::invalid_argument("Unsupported device");
        }, py::arg("device"))
        .def("numpy", [](const Tensor& t) {
            std::vector<float> vec = t.cpu().to_vector();
            std::vector<ssize_t> shape;
            for(auto s : t.shape()) shape.push_back(s);
            return py::array_t<float>(shape, vec.data());
        })
        .def("print", &Tensor::print);

    py::class_<ReLU>(m, "ReLU")
        .def_static("forward", &ReLU::forward)
        .def_static("backward", &ReLU::backward);

    py::class_<Sigmoid>(m, "Sigmoid")
        .def_static("forward", &Sigmoid::forward)
        .def_static("backward", &Sigmoid::backward);

    py::class_<FullyConnected>(m, "FullyConnected")
        .def_static("forward", &FullyConnected::forward)
        .def_static("backward", [](const Tensor& X, const Tensor& W, const Tensor& dY) {
            Tensor dX, dW, db;
            FullyConnected::backward(X, W, dY, dX, dW, db);
            return std::make_tuple(dX, dW, db);
        });

    py::class_<Conv2d>(m, "Conv2d")
        .def_static("forward", &Conv2d::forward)
        .def_static("backward", [](const Tensor& X, const Tensor& W, const Tensor& dY) {
            Tensor dX, dW, db;
            Conv2d::backward(X, W, dY, dX, dW, db);
            return std::make_tuple(dX, dW, db);
        });

    py::class_<MaxPool2x2>(m, "MaxPool2x2")
        .def_static("forward", [](const Tensor& X) {
            IntBuffer mask(0);
            Tensor Y = MaxPool2x2::forward(X, mask);
            return std::make_pair(Y, mask);
        })
        .def_static("backward", &MaxPool2x2::backward);

    py::class_<SoftMax>(m, "SoftMax")
        .def_static("forward", &SoftMax::forward)
        .def_static("cross_entropy", [](const Tensor& logits, const std::vector<int>& labels) {
            Tensor grad;
            float loss = SoftMax::cross_entropy_with_softmax(logits, labels, grad);
            return std::make_pair(loss, grad);
        });
    
    // Factory functions for creating zero-initialized tensors
    m.def("zeros", [](const std::vector<int>& shape, Device device) {
        Tensor t(shape, device);
        t.zeros();
        return t;
    }, py::arg("shape"), py::arg("device")=Device::CPU);
    
    m.def("zeros_like", [](const Tensor& input) {
        Tensor t(input.shape(), input.device());
        t.zeros();
        return t;
    }, py::arg("input"));
        
    // [新增] 暴露优化器
    py::class_<Optimizer>(m, "Optimizer")
        .def_static("sgd_momentum", &Optimizer::sgd_momentum);
}