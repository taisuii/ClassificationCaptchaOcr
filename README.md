
# ClassificationCaptchaOcr

九宫格验证码识别，图像相似度对比，极验九宫格验证码破解，识别

```shell
python run.py
```

文章地址：[https://blog.csdn.net/m0_56516039/article/details/141890708](https://blog.csdn.net/m0_56516039/article/details/141890708)

全套源码+模型部署+极验算法

### 安卓逆向，JS逆向，图像识别，在线接单，QQ: 27788854，wechat: taisuivip
### [telegram: rtais00](https://t.me/rtais00)

微信公众号
![img.png](image/img.png)

# 0x0. 前言
###### 分析验证码
1. 遇到的验证码样式大概是这样的
![验证码样式](https://i-blog.csdnimg.cn/direct/b62096ccaccc4ccd9b7b7cabc0b5d66a.png)

###### 解决方案

 1. 我们可以把每张图片都切出来，然后用CNN提取图像特征
 2. 选用resnet18，再用相似度对比函数对比小图和九张大图的相似度，选出相似度最大的3张就可以了
 3. 因为给了小图，而不是文字，所以我们的特征提取的模型同样可以识别未出现的分类

# 0x1. 准备数据集
###### 获取数据

 1. 一般是一张小图，一张大图，大图切割成9份，切割代码如下
 ```python
import requests
from PIL import Image, ImageFont, ImageDraw, ImageOps
from io import BytesIO
def crop_image(image_bytes, coordinates):
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    grid_width = width // 3
    grid_height = height // 3
    cropped_images = []
    for coord in coordinates:
        y, x = coord
        left = (x - 1) * grid_width
        upper = (y - 1) * grid_height
        right = left + grid_width
        lower = upper + grid_height
        box = (left, upper, right, lower)
        cropped_img = img.crop(box)
        cropped_images.append(cropped_img)
    return cropped_images
# 切割顺序，这里是从左到右，从上到下[x,y]
coordinates = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]
bg_img = requests.get("https://static.geetest.com/captcha_v4/policy/3d0936b11a2c4a65bbb53635e656c780/nine/110394/2024-09-06T00/ed02acd0ac294a41b880d9106240f12a.jpg").content
cropped_images = crop_image(bg_img, coordinates)
# 一个个保存下来
for j, img_crop in enumerate(cropped_images):
    img_crop.save(f"./test_crop/bg{j}.jpg")
```
 2. 保存数据集如下，需要自行分类，可以找人标注，记得要把小图放入
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2304c044256c4811b4f9746cf0f61187.png)
# 0x2. 模型训练
 1. 定义数据集和数据转换，这里要注意，在训练前怎么处理，评估模型的时候也要处理一遍图像再传入模型
 2. 定义模型，这里使用resnet18
 3. 训练代码
 ```python
import torchvision.transforms as transforms
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 训练模型
    for i in range(epoch):
        losses = []

        # 迭代器进度条
        data_loader_tqdm = tqdm(data_loader)

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
                f"This epoch is {i} and it's loss is {loss.item()}, average loss {epoch_loss}"
            )
            # 反向传播：根据当前损失值计算模型参数的梯度
            loss.backward()
            # 使用优化器更新模型参数，根据梯度调整模型参数
            optimizer.step()
        # 每过一个batch就保存一次模型
        torch.save(model.state_dict(), f'model/my_resnet18_{epoch_loss}.pth')
    print(f"completed. Model saved.")
if __name__ == '__main__':
    train(50)
```
- 开跑，训练40个epoch的loss值变化如下
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d474ffde803494b96bc90d27981eee5.png)
- 测试模型，九宫格数字从0到1，分别代表九宫格图片从左到右从上到下的顺序，识别结果正确
- 一个输入为10张图，作为一个batch，拿到10个长度为91的特征向量，把第一个向量与其他九个向量依次对比，选出最大的3个
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/44cf322d6e8c436d96c0b9e5c4dda3c2.png)
# 0x3. 部署为onnx，验证结果
- 导出模型为onnx
 ```python
 from resnet18 import MyResNet18
import torch

def convert():
    # 加载 PyTorch 模型
    model_path = "model/resnet18_38_0.021147585306924.pth"
    model = MyResNet18(num_classes=91)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 生成一个示例输入
    dummy_input = torch.randn(10, 3, 224, 224)
    # 将模型转换为 ONNX 格式
    torch.onnx.export(model, dummy_input, "model/resnet18.onnx", verbose=True)
if __name__ == '__main__':
    convert()
 ```
 - 生产环境下，使用onnx推理50次识别，通过率和速度如下(测试某验官网)，通过率98%
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b59e5abc94674021b6db4e22c2c389f7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3a78464431f749c5984d02cfc0f76d7b.png)
# 0x4. 总结
 - 对于抽象级别较高的图像验证码，卷积神经网络有很好的识别效果
 - github 训练代码: [https://github.com/taisuii/ClassificationCaptchaOcr](https://github.com/taisuii/ClassificationCaptchaOcr)



