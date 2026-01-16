#!/usr/bin/env python3
"""
Test script for verifying the new zeros, zeros_like, and to() bindings.
This script demonstrates the expected usage patterns.
"""

# Note: This test assumes mytensor module is built and importable
# In a CUDA-enabled environment, uncomment the following to test:

# import mytensor
# import numpy as np

# Test 1: zeros() factory function
def test_zeros():
    """Test creating a zero tensor with zeros()"""
    # shape = [2, 3, 4]
    # t = mytensor.zeros(shape, mytensor.Device.GPU)
    # assert t.shape() == shape
    # assert t.device() == mytensor.Device.GPU
    # print("✓ zeros() works correctly")
    print("Test: zeros() - would create tensor with specified shape and device")

# Test 2: zeros_like() factory function
def test_zeros_like():
    """Test creating a zero tensor with zeros_like()"""
    # original = mytensor.Tensor([2, 3], mytensor.Device.GPU)
    # zero_tensor = mytensor.zeros_like(original)
    # assert zero_tensor.shape() == original.shape()
    # assert zero_tensor.device() == original.device()
    # print("✓ zeros_like() works correctly")
    print("Test: zeros_like() - would create zero tensor matching input shape and device")

# Test 3: to() method
def test_to():
    """Test tensor.to(device) method"""
    # t_cpu = mytensor.Tensor([2, 3], mytensor.Device.CPU)
    # t_gpu = t_cpu.to(mytensor.Device.GPU)
    # assert t_gpu.device() == mytensor.Device.GPU
    # 
    # t_back = t_gpu.to(mytensor.Device.CPU)
    # assert t_back.device() == mytensor.Device.CPU
    # print("✓ to() method works correctly")
    print("Test: to() - would transfer tensor between CPU and GPU")

# Test 4: Usage in optimizer (mimics the training script)
def test_optimizer_usage():
    """Test the pattern used in SGD optimizer initialization"""
    # This is how it would be used in the training script:
    # param = mytensor.Tensor([10, 20], mytensor.Device.GPU)
    # velocity = mytensor.zeros_like(param)  # Instead of workaround with fill
    # 
    # Or alternatively:
    # velocity = mytensor.zeros(param.shape()).to(param.device())
    print("Test: Optimizer usage - velocity initialization simplified")

if __name__ == "__main__":
    print("Testing new mytensor bindings...")
    print("=" * 60)
    test_zeros()
    test_zeros_like()
    test_to()
    test_optimizer_usage()
    print("=" * 60)
    print("All tests defined. Actual execution requires built mytensor module.")
    print("\nExpected API:")
    print("  - mytensor.zeros(shape, device=CPU)")
    print("  - mytensor.zeros_like(tensor)")
    print("  - tensor.to(device)")
