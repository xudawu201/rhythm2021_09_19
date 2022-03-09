'''
Author: xudawu
Date: 2022-01-22 19:13:06
LastEditors: xudawu
LastEditTime: 2022-01-24 16:44:37
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
from matplotlib.pyplot import get
import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, image_high,poolKernel_size,in_channels, out_channels, kernel_size, stride,out_features):
        super(CNN,self).__init__()#调用Module的构造函数, super(Linear, self).__init__()
        self.image_high = image_high
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_features = out_features
        self.poolKernel_size = poolKernel_size

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,  #输入图像通道数
                out_channels=self.out_channels,  #卷积产生的通道数,即输出的深度,多少个filter
                kernel_size=self.kernel_size,  #卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3卷积核
                stride=self.stride,  #扫描两个相邻区域之间的步长
                padding=int((kernel_size-stride)/2) #使卷积后图片尺寸不变化
            ),  #卷积层1
            nn.ReLU(),  #激活函数
            nn.MaxPool2d(kernel_size=self.poolKernel_size),  #池化层,降维，减少计算开销。stride默认值是kernel_size

            nn.Conv2d(self.out_channels,self.out_channels*2,self.kernel_size,self.stride,padding=int((kernel_size-stride)/2)),  #卷积层2
            nn.ReLU(),  #激活函数
            nn.MaxPool2d(kernel_size=self.poolKernel_size),
        )
        tensorSize1_float = (self.image_high - self.poolKernel_size)/self.poolKernel_size+1
        tensorSize2_float = (tensorSize1_float - self.poolKernel_size)/self.poolKernel_size+1
        tensorSize_int = int(tensorSize2_float)
        self.out = nn.Linear(in_features=self.out_channels * 2*tensorSize_int*tensorSize_int, out_features=out_features)

    #三维数据展平成2维数据
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 将四维张量转换为二维张量之后，才能作为全连接层的输入
        output = self.out(x)
        return output

#定义网络参数
image_high = 16
poolKernel_size = 3
in_channels = 1
out_channels = 2
kernel_size = 3
stride = 1
out_features = 3
CNN_model = CNN(image_high=image_high,
          poolKernel_size=poolKernel_size,
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=kernel_size,
          stride=stride,
          out_features=out_features)
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
dataset_loader = getDataset(content_tensor, label_tensor, 1)
print(dataset_loader)

#CNN网络训练
lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()  #交叉熵损失函数

print('开始训练')
CNN_model.train()
epoch = 100
for item in range(epoch):
    for input_batch, target_batch in dataset_loader:
        output_tensor = CNN_model(input_batch)
        print('output_tensor:', output_tensor.detach().numpy())
        print(target_batch)
        loss = criterion(output_tensor, target_batch)
        print('loss:{:.4f}'.format(loss.item()))

        #梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #梯度优化
        optimizer.step()