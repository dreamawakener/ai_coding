# Changes Summary: Adding zeros, zeros_like, and to() bindings

## Problem Statement
Users were encountering `AttributeError` when trying to use:
- `mytensor.zeros_like(p.data)` 
- `mytensor.zeros(...)`
- `tensor.to(device)`

These functions were not exposed in the Python bindings, forcing users to use workarounds like manually calling `fill([0.0] * int(v.size()))`.

## Solution
Updated `src/bindings.cpp` to expose three new functions to the Python module:

### 1. Module-level `zeros()` factory function
```cpp
m.def("zeros", [](const std::vector<int>& shape, Device device) {
    Tensor t(shape, device);
    t.zeros();
    return t;
}, py::arg("shape"), py::arg("device")=Device::CPU);
```

**Usage:**
```python
t = mytensor.zeros([2, 3, 4], mytensor.Device.GPU)
```

### 2. Module-level `zeros_like()` factory function
```cpp
m.def("zeros_like", [](const Tensor& input) {
    Tensor t(input.shape(), input.device());
    t.zeros();
    return t;
}, py::arg("input"));
```

**Usage:**
```python
original = mytensor.Tensor([2, 3], mytensor.Device.GPU)
zero_tensor = mytensor.zeros_like(original)
```

### 3. Instance method `to()` on Tensor class
```cpp
.def("to", [](const Tensor& t, Device device) {
    if (device == Device::CPU) return t.cpu();
    else return t.gpu();
}, py::arg("device"))
```

**Usage:**
```python
t_cpu = mytensor.Tensor([2, 3], mytensor.Device.CPU)
t_gpu = t_cpu.to(mytensor.Device.GPU)
```

## Impact on Training Script
The SGD optimizer initialization in `train.py` can now be simplified:

**Before (lines 175-181):**
```python
self.velocities = []
for p in params:
    v = mytensor.Tensor(p.data.shape(), mytensor.Device.GPU)
    v.fill([0.0] * int(v.size()))  # Workaround
    self.velocities.append(v)
```

**After (Option 1 - using zeros_like):**
```python
self.velocities = []
for p in params:
    v = mytensor.zeros_like(p.data)
    self.velocities.append(v)
```

**After (Option 2 - using zeros + to):**
```python
self.velocities = []
for p in params:
    v = mytensor.zeros(p.data.shape()).to(p.data.device())
    self.velocities.append(v)
```

## Files Modified
- `src/bindings.cpp` - Added three new bindings (20 lines added)

## Files Added (for documentation/testing)
- `test_bindings.py` - Test script demonstrating expected usage
- `usage_example.py` - Example showing the before/after comparison

## Testing
The changes have been verified for:
- ✓ C++ syntax correctness (balanced braces, parentheses)
- ✓ Proper integration with existing bindings
- ✓ Consistent API design with existing patterns

**Note:** Full compilation and runtime testing requires a CUDA-enabled environment with proper CUDA toolkit installation. The syntax and structure are correct and ready for compilation.

## Implementation Details
All three functions leverage existing Tensor class methods:
- `zeros()` - Already exists in Tensor class (tensor_lib.h line 36)
- `cpu()` - Already exists in Tensor class  
- `gpu()` - Already exists in Tensor class
- `shape()` - Already exists in Tensor class
- `device()` - Already exists in Tensor class

The bindings simply expose these existing C++ methods to Python in a more user-friendly way.
