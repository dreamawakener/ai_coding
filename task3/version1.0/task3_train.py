import sys
import os
import time
import math
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch # 仅用于数据加载

# [关键] 强制将当前目录加入搜索路径，确保能找到 .so 文件
sys.path.append(os.getcwd())

# ==========================================
# 0. 导入自定义库 (带环境自动修复)
# ==========================================
try:
    import mytensor
except ImportError:
    # 尝试修复 Colab 找不到 PyTorch 共享库的问题
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

print("mytensor loaded successfully.")

# ==========================================
# 1. 自动微分引擎 (Autograd Engine)
# ==========================================

class Variable:
    def __init__(self, tensor, requires_grad=False):
        # 兼容旧版 .so: 如果输入不是 Tensor，则通过 numpy 转换
        if not isinstance(tensor, mytensor.Tensor):
            data_np = np.array(tensor, dtype=np.float32)
            tensor = mytensor.Tensor(data_np, mytensor.Device.GPU)
        self.data = tensor
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None

    def backward(self, grad=None):
        if grad is None:
            grad = mytensor.Tensor(np.array([1.0], dtype=np.float32), mytensor.Device.GPU)
        
        if self.grad is None:
            self.grad = grad
        else:
            # 简化版：直接覆盖梯度 (不支持复杂图的梯��累加)
            self.grad = grad

        if self.creator:
            self.creator.backward(self.grad)

class Function:
    def __call__(self, *args):
        self.inputs = args
        input_tensors = [x.data for x in args]
        self.outputs = self.forward(*input_tensors)
        
        if not isinstance(self.outputs, tuple):
            self.outputs = (self.outputs,)
        
        result = []
        for o in self.outputs:
            if isinstance(o, mytensor.Tensor):
                v = Variable(o, requires_grad=True)
                v.creator = self
                result.append(v)
            else:
                result.append(o)
        return result[0] if len(result) == 1 else result

    def forward(self, *args): raise NotImplementedError
    def backward(self, *grad_outputs): raise NotImplementedError

# ==========================================
# 2. 层定义 (适配 C++ 接口)
# ==========================================

class Conv2dFunction(Function):
    def forward(self, X, W, b):
        self.save_for_backward = (X, W, b)
        return mytensor.Conv2d.forward(X, W, b)
    def backward(self, grad_output):
        X, W, b = self.save_for_backward
        dX, dW, db = mytensor.Conv2d.backward(X, W, grad_output)
        if self.inputs[0].requires_grad: self.inputs[0].backward(dX)
        if self.inputs[1].requires_grad: self.inputs[1].backward(dW)
        if self.inputs[2].requires_grad: self.inputs[2].backward(db)

class LinearFunction(Function):
    def forward(self, X, W, b):
        self.save_for_backward = (X, W, b)
        return mytensor.FullyConnected.forward(X, W, b)
    def backward(self, grad_output):
        X, W, b = self.save_for_backward
        dX, dW, db = mytensor.FullyConnected.backward(X, W, grad_output)
        if self.inputs[0].requires_grad: self.inputs[0].backward(dX)
        if self.inputs[1].requires_grad: self.inputs[1].backward(dW)
        if self.inputs[2].requires_grad: self.inputs[2].backward(db)

class ReLUFunction(Function):
    def forward(self, X):
        self.save_for_backward = (X,)
        return mytensor.ReLU.forward(X)
    def backward(self, grad_output):
        X, = self.save_for_backward
        dX = mytensor.ReLU.backward(X, grad_output)
        if self.inputs[0].requires_grad: self.inputs[0].backward(dX)

class MaxPoolFunction(Function):
    def forward(self, X):
        Y, mask = mytensor.MaxPool2x2.forward(X)
        self.save_for_backward = (X, mask)
        return Y
    def backward(self, grad_output):
        X, mask = self.save_for_backward
        dX = mytensor.MaxPool2x2.backward(X, grad_output, mask)
        if self.inputs[0].requires_grad: self.inputs[0].backward(dX)

class FlattenFunction(Function):
    def forward(self, X):
        self.in_shape = X.shape()
        N = self.in_shape[0]
        size = X.size()
        C_out = size // N
        # 使用 numpy reshape 绕过 C++ 限制
        data_np = X.numpy().reshape(N, int(C_out))
        return mytensor.Tensor(data_np, mytensor.Device.GPU)
    def backward(self, grad_output):
        grad_np = grad_output.numpy().reshape(self.in_shape)
        return mytensor.Tensor(grad_np, mytensor.Device.GPU)

class Conv2d:
    def __init__(self, in_c, out_c):
        # 你的底层实现是 3x3，所以这里只能用 3
        k = 3 
        stdv = 1. / math.sqrt(in_c * k * k)
        w_data = np.random.uniform(-stdv, stdv, (out_c, in_c, k, k)).astype(np.float32)
        b_data = np.random.uniform(-stdv, stdv, (out_c,)).astype(np.float32)
        self.W = Variable(w_data, requires_grad=True)
        self.b = Variable(b_data, requires_grad=True)
    def __call__(self, x):
        return Conv2dFunction()(x, self.W, self.b)

class Linear:
    def __init__(self, in_features, out_features):
        stdv = 1. / math.sqrt(in_features)
        w_data = np.random.uniform(-stdv, stdv, (out_features, in_features)).astype(np.float32)
        b_data = np.random.uniform(-stdv, stdv, (out_features,)).astype(np.float32)
        self.W = Variable(w_data, requires_grad=True)
        self.b = Variable(b_data, requires_grad=True)
    def __call__(self, x):
        return LinearFunction()(x, self.W, self.b)

# ==========================================
# 3. LeNet 模型 (完全对应 PyTorch 结构)
# ==========================================

class LeNet:
    def __init__(self):
        # Conv1: 3 -> 6
        self.conv1 = Conv2d(3, 6) 
        self.pool1 = MaxPoolFunction() 
        
        # Conv2: 6 -> 16
        self.conv2 = Conv2d(6, 16)
        self.pool2 = MaxPoolFunction() 
        
        self.flatten = FlattenFunction()
        
        # 注意: PyTorch LeNet 用 5x5 (无pad) -> 5x5 map -> 16*5*5=400
        # 我们的 C++ 实现是 3x3 (pad=1) -> 8x8 map -> 16*8*8=1024
        # 为了让模型跑起来，这里必须使用 1024
        self.fc1 = Linear(16 * 8 * 8, 120) 
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        x = ReLUFunction()(self.conv1(x)) 
        x = self.pool1(x)
        
        # x = self.pool(F.relu(self.conv2(x)))
        x = ReLUFunction()(self.conv2(x))
        x = self.pool2(x)
        
        x = self.flatten(x)
        
        # x = F.relu(self.fc1(x))
        x = ReLUFunction()(self.fc1(x))
        
        # x = F.relu(self.fc2(x))
        x = ReLUFunction()(self.fc2(x))
        
        x = self.fc3(x)
        return x
    
    def parameters(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, 
                self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]

# ==========================================
# 4. 优化器与损失函数
# ==========================================

class SGD:
    def __init__(self, params, lr=0.001, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data.numpy()) for p in params]
        
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                p_val = p.data.numpy()
                p_grad = p.grad.numpy()
                
                # Momentum update: v = rho * v + g
                self.velocities[i] = self.momentum * self.velocities[i] + p_grad
                # p = p - lr * v
                new_val = p_val - self.lr * self.velocities[i]
                
                p.data = mytensor.Tensor(new_val, mytensor.Device.GPU)
                
    def zero_grad(self):
        for p in self.params:
            p.grad = None

def cross_entropy_loss(logits, labels):
    loss, dlogits = mytensor.SoftMax.cross_entropy(logits.data, labels.tolist())
    logits.grad = dlogits
    return loss

# ==========================================
# 5. 主程序
# ==========================================

def evaluate(net, dataloader):
    correct = 0
    total = 0
    total_loss = 0.0
    steps = 0
    
    for data in dataloader:
        images, labels = data
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        # Inference
        X = Variable(images_np)
        outputs = net.forward(X)
        
        # Calc Loss
        # 注意: 这里会计算梯度但不会 backward
        loss, _ = mytensor.SoftMax.cross_entropy(outputs.data, labels.tolist())
        total_loss += loss
        
        # Calc Accuracy
        pred_y = outputs.data.numpy().argmax(axis=1)
        correct += (pred_y == labels_np).sum()
        total += labels.size(0)
        steps += 1
        
    avg_loss = total_loss / steps if steps > 0 else 0
    acc = 100 * correct / total if total > 0 else 0
    return avg_loss, acc

def main():
    print(f'Using Custom CUDA Implementation (mytensor)')
    
    # 1. 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='../task1/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='../task1/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. 模型与优化器
    net = LeNet()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    start_time = time.time()
    epochs = 10
    
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}')
        running_loss = 0.0
        
        # Training Loop
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            # Forward
            X = Variable(inputs.numpy())
            outputs = net.forward(X)
            
            # Loss & Backward
            loss = cross_entropy_loss(outputs, labels)
            outputs.creator.backward(outputs.grad)
            
            # Optimize
            optimizer.step()
            
            running_loss += loss
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        # Epoch End Evaluation
        print(f"Evaluating epoch {epoch + 1}...")
        train_loss, train_acc = evaluate(net, trainloader)
        test_loss, test_acc = evaluate(net, testloader)
        
        print(f"Epoch {epoch + 1} Finished:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

    print('Finished Training')
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')
    
    # Final Test
    print("Final Test on 10000 images...")
    _, final_acc = evaluate(net, testloader)
    print(f'Accuracy of the network on the 10000 test images: {final_acc} %')

if __name__ == "__main__":
    main()