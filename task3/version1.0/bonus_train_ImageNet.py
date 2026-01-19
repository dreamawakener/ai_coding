import sys, os, time, math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 核心修正：必须先导入 torch，否则找不到 libc10.so
import torch
import mytensor

# ==========================================
# 1. 自动微分引擎 (Python 端)
# ==========================================

class Variable:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.creator = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # 默认梯度为 1.0 (标量 Loss 对自己的梯度)
        if grad is None:
            grad = mytensor.Tensor(self.data.shape(), mytensor.Device.GPU)
            grad.fill([1.0] * int(grad.size()))

        # 梯度累加：self.grad += grad
        if self.grad is None:
            self.grad = grad
        else:
            accumulated_grad = mytensor.Tensor(self.grad.shape(), mytensor.Device.GPU)
            mytensor.Tensor.add(self.grad, grad, accumulated_grad)
            self.grad = accumulated_grad

        # 反向传播给创建者
        if self.creator:
            self.creator.backward(grad)

class Function:
    def __call__(self, *args):
        self.inputs = args
        input_tensors = [x.data for x in args]
        self.outputs = self.forward(*input_tensors)

        if not isinstance(self.outputs, (list, tuple)):
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
# 2. 算子封装 (Binding)
# ==========================================

class Conv2dFunction(Function):
    def __init__(self, stride=1, pad=0):
        self.stride = stride
        self.pad = pad

    def forward(self, X, W, b):
        self.save_for_backward = (X, W, b)
        return mytensor.Conv2d.forward(X, W, b, self.stride, self.pad)

    def backward(self, grad_output):
        X, W, b = self.save_for_backward
        dX, dW, db = mytensor.Conv2d.backward(X, W, grad_output, self.stride, self.pad)

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
        (X,) = self.save_for_backward
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
        size = int(X.size())
        C_out = size // N
        return X.reshape([N, int(C_out)])

    def backward(self, grad_output):
        return grad_output.reshape(self.in_shape)

# ==========================================
# 3. 网络层定义
# ==========================================

class Conv2d:
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        self.pad = padding
        k = kernel_size
        fan_in = in_c * k * k
        stdv = math.sqrt(2.0 / fan_in)

        w_np = np.random.normal(0, stdv, (out_c, in_c, k, k)).astype(np.float32)
        self.W = Variable(mytensor.Tensor(w_np.shape, w_np.ravel().tolist(), mytensor.Device.GPU), requires_grad=True)

        b_np = np.zeros((out_c,), dtype=np.float32)
        self.b = Variable(mytensor.Tensor(b_np.shape, b_np.ravel().tolist(), mytensor.Device.GPU), requires_grad=True)

    def __call__(self, x):
        return Conv2dFunction(stride=1, pad=self.pad)(x, self.W, self.b)

class Linear:
    def __init__(self, in_features, out_features):
        stdv = math.sqrt(2.0 / in_features)

        w_np = np.random.normal(0, stdv, (out_features, in_features)).astype(np.float32)
        self.W = Variable(mytensor.Tensor(w_np.shape, w_np.ravel().tolist(), mytensor.Device.GPU), requires_grad=True)

        b_np = np.zeros((out_features,), dtype=np.float32)
        self.b = Variable(mytensor.Tensor(b_np.shape, b_np.ravel().tolist(), mytensor.Device.GPU), requires_grad=True)

    def __call__(self, x):
        return LinearFunction()(x, self.W, self.b)

# ==========================================
# 4. 模型与优化器
# ==========================================

class TinyImageNetModel:
    def __init__(self, num_classes=200):
        # Input: 3x64x64
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = MaxPoolFunction()
        self.relu = ReLUFunction()
        self.flatten = FlattenFunction()

        # 64 -> 32 -> 16 -> 8
        # Final: 128 * 8 * 8 = 8192
        self.fc1 = Linear(8192, 1024)
        self.fc2 = Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def parameters(self):
        return [
            self.conv1.W, self.conv1.b,
            self.conv2.W, self.conv2.b,
            self.conv3.W, self.conv3.b,
            self.fc1.W, self.fc1.b,
            self.fc2.W, self.fc2.b
        ]

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = []

        for p in params:
            v = mytensor.Tensor(p.data.shape(), mytensor.Device.GPU)
            # 稳健写法：用 fill 0 代替 zeros()，防止 C++ 接口没更新
            size = int(v.size())
            v.fill([0.0] * size)
            self.velocities.append(v)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                mytensor.Optimizer.sgd_momentum(
                    p.data,
                    p.grad,
                    self.velocities[i],
                    self.lr,
                    self.momentum
                )

    def zero_grad(self):
        for p in self.params:
            p.grad = None

def cross_entropy_loss_fn(logits_var, labels_tensor):
    labels_list = labels_tensor.tolist()
    loss_val, probs = mytensor.SoftmaxCrossEntropy.forward(logits_var.data, labels_list)
    grad_logits = mytensor.SoftmaxCrossEntropy.backward(probs, labels_list)
    logits_var.grad = grad_logits
    return loss_val

# ==========================================
# 5. 主程序
# ==========================================

def evaluate(net, dataloader):
    correct = 0
    total = 0
    # print("Evaluating...")
    for inputs, labels in dataloader:
        x_np = inputs.numpy()
        X = Variable(mytensor.Tensor(x_np.shape, x_np.ravel().tolist(), mytensor.Device.GPU), requires_grad=False)

        outputs = net.forward(X)

        logits_np = outputs.data.cpu().to_vector()
        logits_np = np.array(logits_np).reshape(x_np.shape[0], -1)

        predicted = np.argmax(logits_np, axis=1)
        correct += (predicted == labels.numpy()).sum()
        total += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc

def main():
    print(f'Starting Training with Custom C++ Tensor Lib...')

    data_root = './data/tiny-imagenet-200'
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if not os.path.exists(train_dir):
        print(f"Error: Path {train_dir} not found. Please run the data download step above.")
        return

    # 简化的预处理，加快速度
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading Data...")
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    valset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = len(trainset.classes)
    print(f"Data Loaded. Classes: {num_classes}")

    net = TinyImageNetModel(num_classes=num_classes)
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    epochs = 5 # 演示目的，先跑 5 个 epoch

    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()

        print(f"Epoch {epoch+1}/{epochs} running...")

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()

            # 1. 转换数据 (CPU Numpy -> Custom GPU Tensor)
            x_np = inputs.numpy()
            X = Variable(mytensor.Tensor(x_np.shape, x_np.ravel().tolist(), mytensor.Device.GPU))

            # 2. 前向传播
            outputs = net.forward(X)

            # 3. 计算 Loss 和 梯度
            loss = cross_entropy_loss_fn(outputs, labels)

            # 4. 反向传播 (自动微分)
            if outputs.creator:
                outputs.creator.backward(outputs.grad)

            # 5. 更新参数
            optimizer.step()

            running_loss += loss
            if i % 20 == 19:
                end_time = time.time()
                speed = (20 * inputs.shape[0]) / (end_time - start_time + 1e-6)
                print(f'[Epoch {epoch + 1}, Step {i + 1}] Loss: {running_loss / 20:.4f} | Speed: {speed:.1f} img/s')
                running_loss = 0.0
                start_time = time.time()

        val_acc = evaluate(net, valloader)
        print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()