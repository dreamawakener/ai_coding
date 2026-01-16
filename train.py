import sys
import os
import time
import math
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch 

# 强制加载当前目录 .so
sys.path.append(os.getcwd())

try:
    import mytensor
except ImportError:
    try:
        import torch
        torch_lib_path = os.path.dirname(os.path.abspath(torch.__file__)) + "/lib"
        if "LD_LIBRARY_PATH" not in os.environ: os.environ["LD_LIBRARY_PATH"] = torch_lib_path
        else: os.environ["LD_LIBRARY_PATH"] += f":{torch_lib_path}"
        import ctypes
        ctypes.CDLL(os.path.join(torch_lib_path, "libc10.so"))
    except: pass
    import mytensor

print("mytensor loaded successfully.")

# ==========================================
# Autograd Engine
# ==========================================
class Variable:
    def __init__(self, tensor, requires_grad=False):
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
        if self.grad is None: self.grad = grad
        else: self.grad = grad
        if self.creator: self.creator.backward(self.grad)

class Function:
    def __call__(self, *args):
        self.inputs = args
        input_tensors = [x.data for x in args]
        self.outputs = self.forward(*input_tensors)
        if not isinstance(self.outputs, tuple): self.outputs = (self.outputs,)
        result = []
        for o in self.outputs:
            if isinstance(o, mytensor.Tensor):
                v = Variable(o, requires_grad=True)
                v.creator = self
                result.append(v)
            else: result.append(o)
        return result[0] if len(result) == 1 else result
    def forward(self, *args): raise NotImplementedError
    def backward(self, *grad_outputs): raise NotImplementedError

# ==========================================
# Layers
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
        data_np = X.numpy().reshape(N, int(C_out))
        return mytensor.Tensor(data_np, mytensor.Device.GPU)
    def backward(self, grad_output):
        grad_np = grad_output.numpy().reshape(self.in_shape)
        return mytensor.Tensor(grad_np, mytensor.Device.GPU)

class Conv2d:
    def __init__(self, in_c, out_c):
        k = 3 
        stdv = 1. / math.sqrt(in_c * k * k)
        w_data = np.random.uniform(-stdv, stdv, (out_c, in_c, k, k)).astype(np.float32)
        b_data = np.random.uniform(-stdv, stdv, (out_c,)).astype(np.float32)
        self.W = Variable(w_data, requires_grad=True)
        self.b = Variable(b_data, requires_grad=True)
    def __call__(self, x): return Conv2dFunction()(x, self.W, self.b)

class Linear:
    def __init__(self, in_features, out_features):
        stdv = 1. / math.sqrt(in_features)
        w_data = np.random.uniform(-stdv, stdv, (out_features, in_features)).astype(np.float32)
        b_data = np.random.uniform(-stdv, stdv, (out_features,)).astype(np.float32)
        self.W = Variable(w_data, requires_grad=True)
        self.b = Variable(b_data, requires_grad=True)
    def __call__(self, x): return LinearFunction()(x, self.W, self.b)

class LeNet:
    def __init__(self):
        self.conv1 = Conv2d(3, 6) 
        self.pool1 = MaxPoolFunction() 
        self.conv2 = Conv2d(6, 16)
        self.pool2 = MaxPoolFunction() 
        self.flatten = FlattenFunction()
        self.fc1 = Linear(16 * 8 * 8, 120) 
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = ReLUFunction()(self.conv1(x)) 
        x = self.pool1(x)
        x = ReLUFunction()(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = ReLUFunction()(self.fc1(x))
        x = ReLUFunction()(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def parameters(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, 
                self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]

# ==========================================
# [关键修改] GPU 优化器
# ==========================================
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # 初始化速度向量在 GPU 上
        self.velocities = []
        for p in params:
            v = mytensor.Tensor(p.data.shape(), mytensor.Device.GPU)
            # 修改点: 用 fill 代替 zeros (C++ 绑定漏了 zeros)
            # 这一步只在初始化执行一次，不会影响训练速度
            v.fill([0.0] * int(v.size())) 
            self.velocities.append(v)
        
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # [关键修复] 检查梯度是否在 GPU
                # 卷积层和全连接层的 bias 梯度默认在 CPU 计算，必须搬运到 GPU 才能进行加速优化
                # if p.grad.device() == mytensor.Device.CPU:
                #     p.grad = p.grad.gpu()

                # 调用 C++ 内核，全程 GPU 运算
                mytensor.Optimizer.sgd_momentum(p.data, p.grad, self.velocities[i], self.lr, self.momentum)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

def cross_entropy_loss(logits, labels):
    loss, dlogits = mytensor.SoftMax.cross_entropy(logits.data, labels.tolist())
    logits.grad = dlogits
    return loss

def evaluate(net, dataloader):
    correct = 0
    total = 0
    for data in dataloader:
        images, labels = data
        images_np = images.numpy()
        X = Variable(images_np)
        outputs = net.forward(X)
        pred_y = outputs.data.numpy().argmax(axis=1)
        correct += (pred_y == labels.numpy()).sum()
        total += labels.size(0)
    acc = 100 * correct / total if total > 0 else 0
    return acc

def main():
    print(f'Using Custom CUDA Implementation (High Performance Mode)')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='../task1/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../task1/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = LeNet()
    # 策略调整: 初始 LR=0.01 (加速收敛)，后期 decay
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    start_time = time.time()
    epochs = 20 # 增加 epoch 以充分收敛
    
    for epoch in range(epochs):
        # 简单的 LR Decay
        if epoch == 10: 
            optimizer.lr = 0.001
            print("LR Decayed to 0.001")
            
        print(f'Starting epoch {epoch + 1} (LR={optimizer.lr})')
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            X = Variable(inputs.numpy())
            outputs = net.forward(X)
            loss = cross_entropy_loss(outputs, labels)
            outputs.creator.backward(outputs.grad)
            optimizer.step()
            
            running_loss += loss
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        test_acc = evaluate(net, testloader)
        print(f"Epoch {epoch + 1} Finished | Test Acc: {test_acc:.2f}%")

    print('Finished Training')
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')
    _, final_acc = evaluate(net, testloader)
    print(f'Final Accuracy: {final_acc} %')

if __name__ == "__main__":
    main()