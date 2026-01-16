# Example of how the SGD optimizer in train.py can now be simplified

# BEFORE (lines 175-181 in train.py):
# Old workaround because zeros/zeros_like were not available:
"""
self.velocities = []
for p in params:
    v = mytensor.Tensor(p.data.shape(), mytensor.Device.GPU)
    # 修改点: 用 fill 代替 zeros (C++ 绑定漏了 zeros)
    # 这一步只在初始化执行一次，不会影响训练速度
    v.fill([0.0] * int(v.size())) 
    self.velocities.append(v)
"""

# AFTER - with new bindings:
# Option 1: Using zeros_like (most concise)
"""
self.velocities = []
for p in params:
    v = mytensor.zeros_like(p.data)
    self.velocities.append(v)
"""

# Option 2: Using zeros + to (more explicit)
"""
self.velocities = []
for p in params:
    v = mytensor.zeros(p.data.shape()).to(p.data.device())
    self.velocities.append(v)
"""

# The new approach is cleaner, more Pythonic, and aligns with PyTorch's API
