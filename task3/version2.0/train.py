import sys
import os
import time
import math
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch 

# 将当前目录添加到系统路径，以便导入本地模块
sys.path.append(os.getcwd())

# ==========================================
# 0. 导入自定义库 (mytensor)
# ==========================================
try:
    import mytensor
except ImportError:
    # 如果直接导入失败，尝试手动配置动态链接库路径
    try:
        import torch
        # 获取 torch 库的路径，因为自定义库可能依赖 torch 的 C++ 库 (libc10.so)
        torch_lib_path = os.path.dirname(os.path.abspath(torch.__file__)) + "/lib"
        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = torch_lib_path
        else:
            os.environ["LD_LIBRARY_PATH"] += f":{torch_lib_path}"
        import ctypes
        # 显式加载 libc10.so，解决潜在的符号依赖问题
        ctypes.CDLL(os.path.join(torch_lib_path, "libc10.so"))
    except:
        pass
    import mytensor

print("mytensor loaded successfully.")

# ==========================================
# 1. 自动微分引擎 (Autograd Engine)
# ==========================================

class Variable:
    """
    Variable 是对 Tensor 的封装，用于自动求导。
    它不仅存储数据 (data)，还存储梯度 (grad) 以及生成该变量的操作 (creator)。
    """
    def __init__(self, tensor, requires_grad=False):
        if not isinstance(tensor, mytensor.Tensor):
            # 如果输入不是 mytensor.Tensor，则先转换为 float32 的 numpy 数组
            data_np = np.array(tensor, dtype=np.float32)
            
            # 手动获取形状并展平数据，以调用 C++ 侧的 Tensor 构造函数
            # 构造函数签名通常为: Tensor(shape: List[int], data: List[float], device: Device)
            shape = [int(s) for s in data_np.shape]
            data_flat = data_np.flatten().tolist()
            tensor = mytensor.Tensor(shape, data_flat, mytensor.Device.GPU)
            
        self.data = tensor
        self.requires_grad = requires_grad
        self.grad = None     # 用于存储反向传播计算出的梯度
        self.creator = None  # 记录创建该 Variable 的 Function 实例，用于回溯计算图

    def backward(self, grad=None):
        """
        反向传播的入口函数。
        """
        if grad is None:
            # 如果是标量输出（如 loss），默认梯度为 1.0
            grad = mytensor.Tensor([1], [1.0], mytensor.Device.GPU)
        
        # 累积梯度 (此处简化为赋值，标准实现应为累加)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = grad

        # 如果该变量是由某个 Function 创建的，则递归调用父节点的 backward
        if self.creator:
            self.creator.backward(self.grad)

class Function:
    """
    所有可微分操作的基类。
    定义了前向传播 (forward) 和反向传播 (backward) 的接口。
    """
    def __call__(self, *args):
        self.inputs = args
        # 提取输入 Variable 中的 Tensor 数据
        input_tensors = [x.data if isinstance(x, Variable) else x for x in args]
        
        # 执行前向计算
        self.outputs = self.forward(*input_tensors)
        
        # 确保输出是元组形式，方便统一处理
        if not isinstance(self.outputs, tuple):
            self.outputs = (self.outputs,)
        
        # 将输出 Tensor 封装回 Variable，并建立计算图连接 (设置 creator)
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
# 2. 网络层定义 (支持 Padding & Stride)
# ==========================================

class Conv2dFunction(Function):
    """
    2D 卷积操作的封装
    """
    def forward(self, X, W, b, stride, pad):
        # 保存上下文用于反向传播
        self.save_for_backward = (X, W, b)
        self.stride = stride
        self.pad = pad
        # 调用 C++ 后端的前向实现
        return mytensor.Conv2d.forward(X, W, b, stride, pad)

    def backward(self, grad_output):
        X, W, b = self.save_for_backward
        # 调用 C++ 后端的反向实现，计算关于输入、权重和偏置的梯度
        dX, dW, db = mytensor.Conv2d.backward(X, W, grad_output, self.stride, self.pad)
        
        # 将梯度传递给输入变量
        if self.inputs[0].requires_grad: self.inputs[0].backward(dX)
        if self.inputs[1].requires_grad: self.inputs[1].backward(dW)
        if self.inputs[2].requires_grad: self.inputs[2].backward(db)

class LinearFunction(Function):
    """
    全连接层 (Linear/Dense) 操作封装
    """
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
    """
    ReLU 激活函数封装
    """
    def forward(self, X):
        self.save_for_backward = (X,)
        return mytensor.ReLU.forward(X)
    def backward(self, grad_output):
        X, = self.save_for_backward
        dX = mytensor.ReLU.backward(X, grad_output)
        if self.inputs[0].requires_grad: self.inputs[0].backward(dX)

class MaxPoolFunction(Function):
    """
    2x2 最大池化层封装
    """
    def forward(self, X):
        # 前向传播返回池化结果 Y 和 掩码 mask (记录最大值位置用于反向传播)
        Y, mask = mytensor.MaxPool2x2.forward(X)
        self.save_for_backward = (X, mask)
        return Y
    def backward(self, grad_output):
        X, mask = self.save_for_backward
        dX = mytensor.MaxPool2x2.backward(X, grad_output, mask)
        if self.inputs[0].requires_grad: self.inputs[0].backward(dX)

class FlattenFunction(Function):
    """
    展平层，将多维卷积输出展平为向量，用于连接全连接层
    """
    def forward(self, X):
        self.in_shape = X.shape()
        N = self.in_shape[0] # Batch size
        size = X.size()
        C_out = size // N
        # 使用 C++ 绑定进行浅拷贝 reshape
        return X.reshape([N, int(C_out)])

    def backward(self, grad_output):
        # 反向传播时，将梯度 reshape 回原来的 (N, C, H, W) 形状
        return grad_output.reshape(self.in_shape)

class Conv2d:
    """
    卷积层的高级封装，负责初始化权重和偏置
    """
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        k = kernel_size
        # 均匀分布初始化
        stdv = 1. / math.sqrt(in_c * k * k)
        w_data = np.random.uniform(-stdv, stdv, (out_c, in_c, k, k)).astype(np.float32)
        b_data = np.random.uniform(-stdv, stdv, (out_c,)).astype(np.float32)
        
        # 将参数包装为 requires_grad=True 的 Variable
        self.W = Variable(w_data, requires_grad=True)
        self.b = Variable(b_data, requires_grad=True)

    def __call__(self, x):
        return Conv2dFunction()(x, self.W, self.b, self.stride, self.padding)

class Linear:
    """
    全连接层的高级封装，负责初始化权重和偏置
    """
    def __init__(self, in_features, out_features):
        stdv = 1. / math.sqrt(in_features)
        w_data = np.random.uniform(-stdv, stdv, (out_features, in_features)).astype(np.float32)
        b_data = np.random.uniform(-stdv, stdv, (out_features,)).astype(np.float32)
        self.W = Variable(w_data, requires_grad=True)
        self.b = Variable(b_data, requires_grad=True)
    def __call__(self, x):
        return LinearFunction()(x, self.W, self.b)

# ==========================================
# 3. 模型定义 (LeNet)
# ==========================================

class LeNet:
    def __init__(self):
        # 第一层卷积: 输入 3通道 -> 输出 16通道 (32x32 -> 32x32, pad=1)
        self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1) 
        self.pool1 = MaxPoolFunction() # 池化 -> 16x16
        
        # 第二层卷积: 输入 16通道 -> 输出 32通道 (16x16 -> 16x16, pad=1)
        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = MaxPoolFunction() # 池化 -> 8x8
        
        self.flatten = FlattenFunction()
        
        # 全连接层: 输入维度 32 * 8 * 8 = 2048
        self.fc1 = Linear(32 * 8 * 8, 256) 
        self.fc2 = Linear(256, 64)
        self.fc3 = Linear(64, 10) # 输出 10 类

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
        # 收集所有需要更新的参数
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, 
                self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]

# ==========================================
# 4. 优化器与损失函数
# ==========================================

class SGD:
    """
    随机梯度下降优化器，支持动量 (Momentum)
    """
    def __init__(self, params, lr=0.001, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = []
        # 为每个参数初始化对应的速度变量 (Velocity)，用于动量计算
        for p in params:
            v = mytensor.Tensor(p.data.shape(), mytensor.Device.GPU)
            v.zeros()
            self.velocities.append(v)
        
    def step(self):
        # 遍历所有参数进行更新
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # 调用 C++ 优化的 SGD+Momentum 更新核
                mytensor.Optimizer.sgd_momentum(p.data, p.grad, self.velocities[i], self.lr, self.momentum)
                
    def zero_grad(self):
        # 清空梯度
        for p in self.params:
            p.grad = None

def cross_entropy_loss(logits, labels):
    """
    计算 Softmax 交叉熵损失
    """
    # 前向计算：计算 loss 数值和概率分布 probs
    loss, probs = mytensor.SoftmaxCrossEntropy.forward(logits.data, labels.tolist())
    
    # 反向计算：手动计算关于 logits 的梯度
    dlogits = mytensor.SoftmaxCrossEntropy.backward(probs, labels.tolist())
    
    # 将计算出的梯度赋值给 logits 变量，以便开始反向传播链
    logits.grad = dlogits
    
    return loss

# ==========================================
# 5. 训练主循环
# ==========================================

def evaluate(net, dataloader):
    """
    模型评估函数
    """
    correct = 0
    total = 0
    total_loss = 0.0
    steps = 0
    
    for data in dataloader:
        images, labels = data
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        X = Variable(images_np)
        outputs = net.forward(X)
        
        # 计算验证集 Loss
        loss, _ = mytensor.SoftmaxCrossEntropy.forward(outputs.data, labels.tolist())
        total_loss += loss
        
        # 将 GPU Tensor 数据拉回 CPU 并转换为 Numpy 进行准确率计算
        out_cpu = outputs.data.cpu()           # 移动到主机内存
        out_list = out_cpu.to_vector()         # 转换为 Python list
        out_shape = out_cpu.shape()            # 获取形状
        
        out_np = np.array(out_list).reshape(out_shape)
        
        pred_y = out_np.argmax(axis=1)
        correct += (pred_y == labels_np).sum()
        total += labels.size(0)
        steps += 1
        
    avg_loss = total_loss / steps if steps > 0 else 0
    acc = 100 * correct / total if total > 0 else 0
    return avg_loss, acc

def main():
    print(f'Using Custom CUDA Implementation (mytensor)')
    
    # 数据增强：对 CIFAR-10 训练至关重要
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 64
    # 下载并加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root='../task1/data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='../task1/data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = LeNet()
    # 初始学习率 0.01, 动量 0.9
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    start_time = time.time()
    epochs = 30
    
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_start = time.time()
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad() # 清空梯度
            
            # 初始化 Variable，这会将数据上传到 GPU
            X = Variable(inputs.numpy())
            outputs = net.forward(X)
            
            # 计算 Loss 并设置初始梯度
            loss = cross_entropy_loss(outputs, labels)
            
            # 反向传播：从输出层开始回溯
            if outputs.creator:
                outputs.creator.backward(outputs.grad)
            
            # 参数更新
            optimizer.step()
            running_loss += loss

        # 学习率衰减策略
        if epoch == 15:
            optimizer.lr = 0.001
            print("LR decayed to 0.001")
        
        # 评估当前 Epoch 性能
        train_loss, train_acc = evaluate(net, trainloader)
        test_loss, test_acc = evaluate(net, testloader)
        
        print(  f"Epoch {epoch + 1}/{epochs} | T: {time.time()-epoch_start:.1f}s | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

    print('Finished Training')
    print(f'Total time: {time.time() - start_time:.2f}s')

if __name__ == "__main__":
    main()