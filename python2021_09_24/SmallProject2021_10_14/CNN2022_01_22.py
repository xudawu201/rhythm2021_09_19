'''
Author: xudawu
Date: 2022-01-22 19:13:06
LastEditors: xudawu
LastEditTime: 2024-06-29 12:26:30
'''
#建立CNN神经网路
'''
nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
参数：
in_channel:　输入数据的通道数(描述一个像素点)，例RGB图片通道数为3,灰度图像为1；
out_channel: 输出数据的通道数，这个根据模型调整；
kennel_size: 卷积核大小，可以是int，或tuple；kennel_size=2,意味着卷积大小2， kennel_size=（2,3），意味着卷积在第一维度大小为2，在第二维度大小为3；
stride：步长，默认为1，与kennel_size类似，stride=2,意味在所有维度步长为2， stride=（2,3），意味着在第一维度步长为2，意味着在第二维度步长为3；
padding：　零填充

in_channels (int) – Number of channels in the input image
out_channels (int) – Number of channels produced by the convolution
kernel_size (int or tuple) – Size of the convolving kernel
stride (int or tuple, optional) – Stride of the convolution. Default: 1
padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
'''
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels,out_channels,padding,linear_input_size,out_features):
        super(CNN,self).__init__()
        
        # 输入通道数 
        self.in_channels = in_channels
        # 全连接层输入大小
        self.linear_input_size = linear_input_size
        # 第一层卷积输出通道数
        self.out_channels = out_channels
        # 全连接层输出类别数
        self.out_features = out_features
        # 边缘填充0圈数
        self.padding=padding
        
        # 卷积层1
        self.conv1 = nn.Sequential(
            # 卷积层,padding：前向计算时在输入特征图周围添加0的圈数
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, stride=2, padding=self.padding),
            # 归一化层
            nn.BatchNorm2d(self.out_channels),
            # 池化层
            nn.MaxPool2d(kernel_size=5, stride=2)
        )

        # 卷积层2
        self.conv2 = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels*2, kernel_size=4, stride=2, padding=self.padding),
            # 归一化层
            nn.BatchNorm2d(self.out_channels*2),
            # 池化层
            nn.MaxPool2d(kernel_size=4, stride=2)
        )

        # 卷积层3
        self.conv3 = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=self.out_channels*4, kernel_size=3, stride=1, padding=self.padding),
            # 归一化层
            nn.BatchNorm2d(self.out_channels*4),
            # 池化层
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        # 展平层
        self.flatten = torch.nn.Flatten()
        # 激活层1
        self.gelu = nn.GELU()
        # 全连接层1
        self.fc1 = nn.Linear(self.linear_input_size, self.out_features)

    def forward(self,x):
        # 多层卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 展平层
        x=self.flatten(x)
        # 激活层
        x = self.gelu(x)
        # 全连接层1
        x = self.fc1(x)
        return x

# 定义网络参数
# 输入图像通道数
in_channels = 1
# 第一层卷积输出通道数
out_channels=2
# 边缘填充0圈数
padding_size=4
# 全连接层输入大小
linear_input_size=200
# 全连接层输出类别数
out_features = 3
# 实例化cnn网络
CNN_model = CNN(in_channels,out_channels,padding_size,linear_input_size,out_features)
print(CNN_model)  #打印结构

#CNN网络测试
temp1_list=[]
content_list = []
for i in range(256):
    temp1_list.append(i)
content_list.append(temp1_list)

temp2_list = []
for i in range(255,-1,-1):
    temp2_list.append(i)
content_list.append(temp2_list)

temp3_list = []
for i in range(101):
    temp3_list.append(i)
for i in range(255,100,-1):
    temp3_list.append(i)
content_list.append(temp3_list)

print('content_list')
print(len(content_list))
label_list=[[1,0,0],[0,1,0],[0,0,1]]
content_tensor=torch.FloatTensor(content_list[0])
print('content',content_tensor.shape)
a = content_tensor.reshape(-1,1,16, 16)#16*16的4维数组
print('a',a.shape)
output = CNN_model(a)
print('output:')
print(output)


#构建数据集
def getDataset(input_tensor, labels_tensor, batch_size):
    from torch.utils import data
    #包装数据存入数据集
    dataset = data.TensorDataset(input_tensor, labels_tensor)
    #从数据集分批次获取一些数据
    dataset_loader = data.DataLoader(dataset, batch_size, shuffle=False)
    return dataset_loader

content_tensor = torch.FloatTensor(content_list).reshape(-1, 1, 16, 16)
print('- '*10,'分割线','- '*10)
print('content',content_tensor.shape)
label_tensor = torch.FloatTensor(label_list)
dataset_loader = getDataset(content_tensor, label_tensor, 8)
print(dataset_loader)

#CNN网络训练
lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()  #交叉熵损失函数

print('开始训练')
CNN_model.train()
epoch = 200
for item in range(epoch):
    for input_batch, target_batch in dataset_loader:
        output_tensor = CNN_model(input_batch)
        # print('output_tensor:', output_tensor.detach().numpy())
        # print(target_batch)
        loss = criterion(output_tensor, target_batch)
        print('loss:{:.4f}'.format(loss.item()))

        #反向传播
        loss.backward()
        #梯度优化
        optimizer.step()
        #梯度清零
        optimizer.zero_grad()
