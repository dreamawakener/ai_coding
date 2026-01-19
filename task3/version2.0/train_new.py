import sys, os, time, math, numpy as np
import torchvision, torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 加载 C++ 扩展
import mytensor

# ==========================================
# 1. Dataset (保持不变)
# ==========================================
class TinyImageNetValDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        
        anno_path = os.path.join(root, 'val_annotations.txt')
        img_dir = os.path.join(root, 'images')
        
        if not os.path.exists(anno_path):
            raise RuntimeError(f"Annotation file not found: {anno_path}")

        with open(anno_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                filename = parts[0]
                synset = parts[1]
                if synset in class_to_idx:
                    self.images.append(os.path.join(img_dir, filename))
                    self.labels.append(class_to_idx[synset])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================================
# Autograd Engine (保持不变)
# ==========================================
class Variable:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.creator = None

    def backward(self, grad=None):
        if grad is None:
            grad = mytensor.Tensor([1], self.data.device())
            grad.fill([1.0])
        
        if self.grad is None:
            self.grad = grad
        else:
            accumulated_grad = mytensor.Tensor(self.grad.shape(), self.data.device())
            mytensor.Tensor.add(self.grad, grad, accumulated_grad)
            self.grad = accumulated_grad
            
        if self.creator:
            self.creator.backward(grad)

class Function:
    def __call__(self, *args):
        self.inputs = args
        input_tensors = [x.data for x in args]
        self.outputs = self.forward(*input_tensors)
        if not isinstance(self.outputs, tuple): self.outputs = (self.outputs,)

        result = []
        for o in self.outputs:
            if isinstance(o, mytensor.Tensor):
                # 如果输入不需要梯度，输出通常也不需要（除非手动指定）
                req_grad = any(x.requires_grad for x in args)
                v = Variable(o, requires_grad=req_grad)
                if req_grad:
                    v.creator = self
                result.append(v)
            else:
                result.append(o)
        return result[0] if len(result) == 1 else result

    def forward(self, *args): raise NotImplementedError
    def backward(self, *grad_outputs): raise NotImplementedError

# ==========================================
# Wrappers (保持不变)
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
        return X.reshape([N, int(C_out)])
    
    def backward(self, grad_output):
        dX = grad_output.reshape(self.in_shape)
        if self.inputs[0].requires_grad:
            self.inputs[0].backward(dX)

def tensor_from_numpy(np_array, requires_grad=False):
    # 辅助函数：快速创建 Variable
    return Variable(
        mytensor.Tensor(np_array.shape, np_array.flatten().tolist(), mytensor.Device.GPU),
        requires_grad=requires_grad
    )

# ==========================================
# Model Definitions
# ==========================================
class Conv2d:
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        k = kernel_size
        self.pad = padding
        # He Initialization (Kaiming Normal) - 适合 ReLU
        stdv = math.sqrt(2. / (in_c * k * k))
        w_data = np.random.randn(out_c, in_c, k, k).astype(np.float32) * stdv
        b_data = np.zeros(out_c, dtype=np.float32)
        self.W = tensor_from_numpy(w_data, requires_grad=True)
        self.b = tensor_from_numpy(b_data, requires_grad=True)
    def __call__(self, x): return Conv2dFunction(stride=1, pad=self.pad)(x, self.W, self.b)

class Linear:
    def __init__(self, in_features, out_features):
        # He Initialization for Linear
        stdv = math.sqrt(2. / in_features)
        w_data = np.random.randn(out_features, in_features).astype(np.float32) * stdv
        b_data = np.zeros(out_features, dtype=np.float32)
        self.W = tensor_from_numpy(w_data, requires_grad=True)
        self.b = tensor_from_numpy(b_data, requires_grad=True)
    def __call__(self, x): return LinearFunction()(x, self.W, self.b)

# ----------------------------------------------
# 改进后的模型：VGG-Style (Deeper & Wider)
# ----------------------------------------------
class ImageNetModel:
    def __init__(self, num_classes=200):
        # Input: 3 x 64 x 64
        
        # Block 1: [64, 64] -> MaxPool -> 32x32
        self.conv1_1 = Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = MaxPoolFunction()
        
        # Block 2: [128, 128] -> MaxPool -> 16x16
        self.conv2_1 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = MaxPoolFunction()
        
        # Block 3: [256, 256] -> MaxPool -> 8x8
        self.conv3_1 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = MaxPoolFunction()
        
        self.flatten = FlattenFunction()
        
        # FC Layers
        # Input features: 256 channels * 8 * 8 = 16384
        self.fc1 = Linear(16384, 1024)
        self.fc2 = Linear(1024, 1024)
        self.fc3 = Linear(1024, num_classes)

    def forward(self, x):
        # Block 1
        x = ReLUFunction()(self.conv1_1(x))
        x = ReLUFunction()(self.conv1_2(x))
        x = self.pool1(x)
        
        # Block 2
        x = ReLUFunction()(self.conv2_1(x))
        x = ReLUFunction()(self.conv2_2(x))
        x = self.pool2(x)
        
        # Block 3
        x = ReLUFunction()(self.conv3_1(x))
        x = ReLUFunction()(self.conv3_2(x))
        x = self.pool3(x)
        
        # Head
        x = self.flatten(x)
        x = ReLUFunction()(self.fc1(x))
        x = ReLUFunction()(self.fc2(x))
        x = self.fc3(x) # Logits
        return x

    def parameters(self):
        return [
            self.conv1_1.W, self.conv1_1.b, self.conv1_2.W, self.conv1_2.b,
            self.conv2_1.W, self.conv2_1.b, self.conv2_2.W, self.conv2_2.b,
            self.conv3_1.W, self.conv3_1.b, self.conv3_2.W, self.conv3_2.b,
            self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b
        ]

# ==========================================
# Optimizer & Loss
# ==========================================
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = []
        for p in params:
            v = mytensor.Tensor(p.data.shape(), mytensor.Device.GPU)
            v.zeros()
            self.velocities.append(v)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                mytensor.Optimizer.sgd_momentum(p.data, p.grad, self.velocities[i], self.lr, self.momentum)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

def cross_entropy_loss(logits, labels):
    loss_val, probs = mytensor.SoftmaxCrossEntropy.forward(logits.data, labels.tolist())
    dlogits = mytensor.SoftmaxCrossEntropy.backward(probs, labels.tolist())
    logits.grad = dlogits
    return loss_val

def evaluate(net, dataloader):
    correct = 0
    total = 0
    print("Evaluating...", end='', flush=True)
    for data in dataloader:
        images, labels = data
        # 重要：验证阶段 requires_grad=False，避免显存爆炸和无用计算
        X = tensor_from_numpy(images.numpy(), requires_grad=False)
        
        # Forward
        outputs = net.forward(X)
        
        pred_y_all = outputs.data.cpu().to_vector()
        pred_y_all = np.array(pred_y_all).reshape(images.shape[0], -1)
        pred_y = pred_y_all.argmax(axis=1)
        
        correct += (pred_y == labels.numpy()).sum()
        total += labels.size(0)
    print(" Done.")
    return 100 * correct / total if total > 0 else 0

# ==========================================
# Main Train Loop
# ==========================================
def main_imagenet():
    print(f'Starting ImageNet Training (VGG-Style)')

    data_root = './data/tiny-imagenet-200'
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if not os.path.exists(train_dir):
        print(f"Error: Path {train_dir} not found.")
        return

    # 增强的数据增广
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4), # 防止过拟合的关键
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 增加颜色扰动
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading Train Set...")
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4) # 增加 workers 加速

    print("Loading Validation Set...")
    valset = TinyImageNetValDataset(root=val_dir, class_to_idx=trainset.class_to_idx, transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    num_classes = len(trainset.classes)
    print(f"Detected {num_classes} classes.")
    
    net = ImageNetModel(num_classes=num_classes)
    # 由于没有 BN，初始学习率不能太高，但 Momentum 需要保持
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    epochs = 50 

    best_acc = 0.0

    for epoch in range(epochs):
        # 调整学习率策略：更平滑的衰减
        if epoch == 20:
            optimizer.lr = 0.005
            print(f"LR Decayed to {optimizer.lr}")
        elif epoch == 35:
            optimizer.lr = 0.001
            print(f"LR Decayed to {optimizer.lr}")
        elif epoch == 45:
            optimizer.lr = 0.0001
            print(f"LR Decayed to {optimizer.lr}")

        print(f'Epoch {epoch + 1}/{epochs} | LR: {optimizer.lr}')
        net_train_step(net, trainloader, optimizer)

        val_acc = evaluate(net, valloader)
        print(f"Validation Accuracy: {val_acc:.2f}% (Best: {best_acc:.2f}%)")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 可以保存模型权重（如果有 save 功能）

def net_train_step(net, loader, optimizer):
    running_loss = 0.0
    start = time.time()
    for i, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()

        # 训练时 requires_grad=False (Tensor本身不需要，权重需要)
        # Variable 内部会自动构建计算图
        X = tensor_from_numpy(inputs.numpy(), requires_grad=False)
        outputs = net.forward(X)
        loss = cross_entropy_loss(outputs, labels)

        if outputs.creator:
            outputs.creator.backward(outputs.grad)

        optimizer.step()

        running_loss += loss
        if i % 50 == 49:
            fps = (50 * loader.batch_size) / (time.time() - start)
            print(f'[Step {i + 1}] loss: {running_loss / 50:.3f} | speed: {fps:.2f} img/s')
            running_loss = 0.0
            start = time.time()

if __name__ == "__main__":
    main_imagenet()