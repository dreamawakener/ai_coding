import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os

# ==========================================
# 1. 模型定义 (Same Architecture)
# ==========================================
class PyTorchLeNet(nn.Module):
    def __init__(self):
        super(PyTorchLeNet, self).__init__()
        # 对应自定义代码: self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1)
        # padding=1 使得 32x32 -> 32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # 对应自定义代码: self.pool1 = MaxPoolFunction() (2x2, stride 2)
        # 32x32 -> 16x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 对应自定义代码: self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        # 16x16 -> 16x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Pool2: 16x16 -> 8x8
        
        # Flatten: 32 channels * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC Layers
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x

# ==========================================
# 2. 评估函数
# ==========================================
def evaluate(net, dataloader, device, criterion):
    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            steps += 1
            
    avg_loss = total_loss / steps if steps > 0 else 0
    acc = 100 * correct / total if total > 0 else 0
    return avg_loss, acc

# ==========================================
# 3. 训练主函数
# ==========================================
def main():
    print(f'Using Standard PyTorch Implementation')
    
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据增强 (保持一致)
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
    trainset = torchvision.datasets.CIFAR10(root='../task1/data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='../task1/data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化网络
    net = PyTorchLeNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 初始 LR 0.01, Momentum 0.9
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    start_time = time.time()
    epochs = 30
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        # 手动调整学习率 (对应 if epoch == 15)
        if epoch == 15:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
            print("LR decayed to 0.001")
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()

        # 评估
        train_loss, train_acc = evaluate(net, trainloader, device, criterion)
        test_loss, test_acc = evaluate(net, testloader, device, criterion)
        
        # 保持完全相同的打印格式
        print(  f"Epoch {epoch + 1}/{epochs} | T: {time.time()-epoch_start:.1f}s | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

    print('Finished Training')
    print(f'Total time: {time.time() - start_time:.2f}s')

if __name__ == "__main__":
    main()