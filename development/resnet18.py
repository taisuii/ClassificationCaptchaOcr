import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# 定义数据转换
data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),  # 标准化图像
    ]
)


# 定义数据集
class CustomDataset:
    def __init__(self, data_dir):
        self.dataset = ImageFolder(root=data_dir, transform=data_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


class MyResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super(MyResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)  # 修改这里的输入大小为512

    def forward(self, x):
        return self.resnet(x)


def train(epoch):
    print("judge the cuda: " + str(torch.version.cuda))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("this train use devices: " + str(device))

    data_dir = "dataset"
    # 自定义数据集实例
    custom_dataset = CustomDataset(data_dir)
    # 数据加载器
    batch_size = 64
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型 num_classes就是目录下的子文件夹数目，每个子文件夹对应一个分类，模型输出的向量长度也是这个长度
    model = MyResNet18(num_classes=91)
    model.to(device)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_losses = []
    # 训练模型
    for i in range(epoch):
        losses = []

        # 迭代器进度条
        data_loader_tqdm = tqdm(data_loader)

        epoch_loss = 0
        for inputs, labels in data_loader_tqdm:
            # 将输入数据和标签传输到指定的计算设备（如 GPU 或 CPU）
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度更新之前将所有模型参数的梯度置为零，防止梯度累积
            optimizer.zero_grad()

            # 前向传播：将输入数据传入模型，计算输出
            outputs = model(inputs)

            # 根据模型的输出和实际标签计算损失值
            loss = criterion(outputs, labels)

            # 将当前批次的损失值记录到 losses 列表中，以便后续计算平均损失
            losses.append(loss.item())
            epoch_loss = np.mean(losses)
            data_loader_tqdm.set_description(
                f"This epoch is {str(i + 1)} and it's loss is {loss.item()}, average loss {epoch_loss}"
            )

            # 反向传播：根据当前损失值计算模型参数的梯度
            loss.backward()
            # 使用优化器更新模型参数，根据梯度调整模型参数
            optimizer.step()
        epoch_losses.append(epoch_loss)
        # 每过一个batch就保存一次模型
        torch.save(model.state_dict(), f'model/resnet18_{str(i + 1)}_{epoch_loss}.pth')

    # loss 变化绘制代码
    data = np.array(epoch_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f"{epoch} epoch loss change")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    # 显示图像
    plt.show()
    print(f"completed. Model saved.")


if __name__ == '__main__':
    train(40)
