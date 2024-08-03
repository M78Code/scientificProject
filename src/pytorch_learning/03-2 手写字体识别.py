# 1 加载必要的库
import torch
import torch.nn as nn  # 神经网络库
import torch.nn.functional as F
import torch.optim as optim  # 优化器
from torchvision import datasets, transforms  # torchvision操作数据库

from torch.utils.data import DataLoader

# 2 定义超参数
BATCH_SIZE = 256  # 每批处理的数据

# 要想像使用服务器的GPU上进行深度学习加速，就需要将模型放到GPU上，在服务器中这个操作是通过
# device = torch.device("cuda:0")
# model = model.to(device)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 是否用GPU还是CPU训练(非Mac系的电脑可以这样写)
# 使用Mac系的M芯片进行加速 https://juejin.cn/post/7137891506777489416
DEVICE_MACBOOK = torch.device("mps")
EPOCHS = 100  # 训练数据集的轮次

# 3 构建pipeline, 对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 对图片进行正则化,作用，当模型出现过拟合现象时，降低模型复杂度
])

# 4 下载，加载数据

# 下载数据集
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

# 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # shuffle 打乱数据
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# 5 构建网络模型（***）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层定义
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1: 灰度图片的通道， 10: 输出通道， 5: kernel
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10: 输入通道， 20: 输出通道， 3: kernel
        # 全连接层定义
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 20*10*10: 输入通道， 500: 输出通道
        self.fc2 = nn.Linear(500, 10)  # 500: 输入通道， 10: 输出通道

    # 反向传播
    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)  # 输入: batch*1*28*28 , 输出: batch*10*24*24 ( 28 - 5 + 1 = 24 )
        x = F.relu(x)  # 保持shape不变， 输出: batch*10*24*24
        x = F.max_pool2d(x, 2, 2)  # 输入: batch*10*24*24 输出: batch*10*12*12

        x = self.conv2(x)  # 输入: batch*10*12*12 输出: batch*20*10*10 ( 12 - 3 + 1 = 10 )
        x = F.relu(x)

        x = x.view(input_size, -1)  # 拉平, -1 自动计算维度， 20*10*10= 2000

        x = self.fc1(x)  # 输入: batch*2000 输出: batch*500
        x = F.relu(x)  # 保持shape不变

        x = self.fc2(x)  # 输入: batch*500 输出: batch * 10

        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值

        return output


# 6 定义优化器
model = Net().to(DEVICE_MACBOOK)

optimizer = optim.Adam(model.parameters())


# 7 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到DEVICE上去
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失（损失就是差距，和真实值的差距）
        loss = F.cross_entropy(output, target)  # (cross_entropy：适合交叉熵函数，如果是二分类问题，就用sigmod函数)
        # 找到概率真最大的下标
        # pred = output.max(1, keepdim=True) # 或者 pred = output.argmax(dim=1)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))
