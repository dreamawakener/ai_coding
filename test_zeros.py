#!/usr/bin/env python3
"""Test script to verify the zeros() method is properly exposed to Python."""

import sys
import os
import numpy as np

# Force load current directory .so
sys.path.append(os.getcwd())

try:
    import mytensor
except ImportError:
    try:
        import torch
        torch_lib_path = os.path.dirname(os.path.abspath(torch.__file__)) + "/lib"
        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = torch_lib_path
        else:
            os.environ["LD_LIBRARY_PATH"] += f":{torch_lib_path}"
        import ctypes
        ctypes.CDLL(os.path.join(torch_lib_path, "libc10.so"))
    except:
        pass
    import mytensor

def test_zeros_cpu():
    """Test zeros() method on CPU."""
    print("Testing zeros() on CPU...")
    t = mytensor.Tensor([2, 3], mytensor.Device.CPU)
    t.fill([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Fill with non-zero values
    
    # Call zeros to reset
    t.zeros()
    
    # Convert to numpy and verify all zeros
    vec = t.to_vector()
    assert all(v == 0.0 for v in vec), f"Expected all zeros, got {vec}"
    print(f"✓ CPU test passed: all {len(vec)} elements are zero")
    return True

def test_zeros_gpu():
    """Test zeros() method on GPU."""
    print("Testing zeros() on GPU...")
    t = mytensor.Tensor([3, 4], mytensor.Device.GPU)
    t.fill([1.0] * 12)  # Fill with non-zero values
    
    # Call zeros to reset
    t.zeros()
    
    # Convert to CPU and numpy and verify all zeros
    vec = t.cpu().to_vector()
    assert all(v == 0.0 for v in vec), f"Expected all zeros, got {vec}"
    print(f"✓ GPU test passed: all {len(vec)} elements are zero")
    return True

def test_zeros_initialization():
    """Test zeros() can be used for initialization."""
    print("Testing zeros() for initialization pattern...")
    shape = [5, 10]
    t = mytensor.Tensor(shape, mytensor.Device.CPU)
    t.zeros()  # Initialize to zeros
    
    vec = t.to_vector()
    assert len(vec) == 50, f"Expected 50 elements, got {len(vec)}"
    assert all(v == 0.0 for v in vec), "Expected all zeros"
    print(f"✓ Initialization test passed: tensor of shape {shape} initialized to zeros")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing zeros() method binding")
    print("=" * 60)
    
    try:
        all_passed = True
        all_passed &= test_zeros_cpu()
        all_passed &= test_zeros_gpu()
        all_passed &= test_zeros_initialization()
        
        if all_passed:
            print("\n" + "=" * 60)
            print("✓ All tests passed!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
