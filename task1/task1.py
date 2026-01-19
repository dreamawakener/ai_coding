import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 定义一个更适合 CIFAR-10 的 VGG 风格网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.25)

        # FC
        self.flatten = nn.Flatten()
        # 经过两次池化，32x32 -> 16x16 -> 8x8。通道数为128
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        # Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)

        # FC
        x = self.flatten(x)
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def evaluate_accuracy(net, dataloader, device):
    """计算模型在给定数据集上的准确率"""
    correct = 0
    total = 0
    net.eval() # 切换到评估模式
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # 1. 硬件配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 开启 cudnn benchmark，针对固定输入尺寸可以加速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 2. 数据预处理与增强
    # 使用 CIFAR-10 的统计均值和方差
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # 训练集：增加数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 随机裁剪
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # 测试集：仅标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # 3. 数据加载 (增大 Batch Size)
    batch_size = 128 
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True)

    classes = ( 'plane', 'car', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck')

    # 4. 初始化模型
    net = CNN().to(device)

    # 5. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    # 使用 SGD + Momentum
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 学习率调度器：在第 10 和 20 个 epoch 衰减学习率
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    print(f'Start Training (Batch size: {batch_size})...')
    start_time = time.time()
    
    epochs = 25 # 适当增加 Epoch
    
    for epoch in range(epochs):
        net.train() # 切换回训练模式
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 每个 Epoch 结束时计算一次测试集准确率
        train_acc = evaluate_accuracy(net, trainloader, device)
        test_acc = evaluate_accuracy(net, testloader, device)
        
        avg_loss = running_loss / len(trainloader)
        
        # 输出训练信息
        print(  f'Epoch [{epoch + 1}/{epochs}] '
                f'LR: {current_lr:.4f} | '
                f'Loss: {avg_loss:.4f} | '
                f'Train Acc: {train_acc:.2f}% | '
                f'Test Acc: {test_acc:.2f}%')

    end_time = time.time()
    print(f'Finished Training. Total time: {end_time - start_time:.2f} seconds')

    # 保存模型
    torch.save(net.state_dict(), './cifar_optimized_net.pth')

    # 6. 最终各类别准确率详情
    print("\nClass-wise Accuracy:")
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.1f} %')

if __name__ == '__main__':
    main()