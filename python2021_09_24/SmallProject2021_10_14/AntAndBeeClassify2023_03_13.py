'''
Author: xudawu
Date: 2023-03-13 13:36:10
LastEditors: xudawu
LastEditTime: 2023-03-23 20:15:26
'''
'''
文件结构
DataSet
    ——hymenoptera_data
        ——train
            xxx.jpg
            xxx.jpg
        ——val
            xxx.jpg
            xxx.jpg
'''
# 获取图片文件夹和类别
from torchvision.datasets import ImageFolder
# 图像调整
from torchvision import transforms
import torch
# 图像统一调整
transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])
dataset_train=ImageFolder('AntAndBeeClassify2023_03_13\\DataSet\\hymenoptera_data\\train',transform=transform)
print(len(dataset_train))
print(dataset_train[0])
# 返回类别
print(dataset_train.classes)
# 返回类别及索引
print(dataset_train.class_to_idx) 
# 是一个元组
print(dataset_train[128][0],dataset_train[128][1])
# 获得图片路径名和类别
# print(dataset_train.imgs)
# 获得单个图片路径名和类别
print(dataset_train.imgs[0])

# 加载数据
# torch自带的标准数据集加载函数
from torch.utils.data import DataLoader
dataloader_train=DataLoader(dataset_train,batch_size=1,shuffle=True)
# 使用提供模型
# 构造网络
import torchvision.models as models
from torch import nn

# 不需要预训练权重,pretrained默认也为False
# 改变全连接层数据
# 使用resnet()
# NN_model=models.resnet18()
# 使用pytorch2.0提高训练速度
# NN_model = torch.compile(NN_model)
# in_features=NN_model.fc.in_features
# 将最后的全连接改为(36，2)
# NN_model.fc=nn.Sequential(nn.Linear(in_features,36),
#                           nn.Linear(36,2))

# 使用densenet()
# NN_model=models.densenet121()
# 使用pytorch2.0提高训练速度
# NN_model = torch.compile(NN_model)
# in_features=NN_model.classifier.in_features

# 使用vgg19()
# NN_model=models.vgg19()
# 使用pytorch2.0提高训练速度
# NN_model = torch.compile(NN_model)
# 第7个为全连接层
# in_features=NN_model.classifier[6].in_features

# 设置输出类别
# NN_model.fc=nn.Linear(in_features, 2)

# 自己构建
import torch
from torch import nn
'''
1.
conv2d_shape:
input(N,C_in,H_in,W_in) # batch, channel , height , width
output(N,C_out,H_out,W_out)
C_out=out_channels
H_out/W_out = (H_out/W_out - kennel_size + 2*padding) / stride + 1
2.
MaxPool2d_shape:
同conv2d
3.
Linear_shape
input(N,C_in,H_in,W_in)
Linear(in_features,out_features)
in_features=C_in*H_in*W_in
'''
class CNN(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,stride,out_features):
        super(CNN,self).__init__()#调用Module的构造函数, super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_features = out_features
        
        # 第一层卷积
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride)
        
        # 全连接层
        self.feature_size=36
        self.linear1 = nn.Linear(in_features=260100,out_features=self.feature_size)
        self.linear2 = nn.Linear(in_features=self.feature_size,out_features=self.out_features)

    # 前馈
    def forward(self,x):
        x = self.conv1(x)
        # 将四维张量转换为二维张量之后，才能作为全连接层的输入
        x = x.view(x.size(0), -1)
        output = self.linear1(x)
        output = self.linear2(output)
        return output

# 实例化网络模型
# 参数
in_channels = 3
out_channels = 1
kernel_size = 3
stride = 1
# padding=int((kernel_size - stride) / 2)
out_features=2
# 实例化
NN_model=CNN(in_channels,out_channels,kernel_size,stride,out_features)
# 使用pytorch2.0提高训练速度
# NN_model = torch.compile(NN_model)

# 测试一个图像输出,使用pytorch框架的网络模型时要注意维度
# x=dataset_train[0][0]
# print(x.shape)
# print(x)
# y=NN_model(x)
# print(y.shape)
# print(y)

# 损失函数和优化器
criterion=nn.CrossEntropyLoss()
lr=0.01
optimizer = torch.optim.Adam(NN_model.parameters(), lr=lr)

# 进行训练
import time
from datetime import timedelta

# 获取已使用时间
def getTimeUsed(startTime_time):
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    return timedelta(seconds=int(round(time_dif)))

# 动态绘图
def dynamicDraw(x_list,y_list,xd,yd):
    plt.cla() # 清除之前的绘图
    x_list.append(xd)  # 添加x轴数据
    y_list.append(yd)  # 添加y轴数据
    # ls：线条风格 (linestyle)
    plt.plot(x_list, y_list,color='b',ls='-',label='line') # 拟合函数
    plt.pause(0.01)#暂停多少秒查看图像
    return x_list,y_list
    
import matplotlib.pyplot as plt

# 将模型迁移到gpu
NN_model=NN_model.cuda()
criterion=criterion.cuda()

epoch=20
for step in range(epoch):
    print('epoch:',step+1)
    starTime_time=time.time()
    totalLoss=0.0
    plt.ion() # 开启交互模式
    x_list=[] # 训练x轴历史数据
    y_list=[] # 训练y轴历史数据
    countStep=0
    for data in dataloader_train:
        # 取出训练数据
        imgs,targets=data
        # 转到GPU上
        imgs=imgs.cuda()
        targets=targets.cuda()
        # 获得网络输出
        outPut_tensor=NN_model(imgs)
        loss=criterion(outPut_tensor,targets)
        # print('loss:',f'{loss.item():.4f}')
        totalLoss=totalLoss+loss
        
        #梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #梯度优化
        optimizer.step()

        # 绘图
        # 训练x和y信息
        yTrain=loss.tolist()
        # x_list,y_list=dynamicDraw(x_list,y_list,countStep,yTrain)
        countStep=countStep+1
    
    print('loss:',f'{totalLoss:.4f}','用时:', getTimeUsed(starTime_time))

# 训练完成
print('训练完成,开始验证')

# 验证
# 测试一个图像输出
# x=dataset_train[0][0]
# y=NN_model(x)
# print('cnn:',y,'y最大值索引:',torch.argmax(y).tolist(),' real:',dataset_train[0][1])

# x=dataset_train[2][0]
# y=NN_model(x)
# print('cnn:',y,'y最大值索引:',torch.argmax(y).tolist(),' real:',dataset_train[2][1])

# x=dataset_train[128][0]
# y=NN_model(x)
# print('cnn:',y,'y最大值索引:',torch.argmax(y).tolist(),' real:',dataset_train[128][1])

# x=dataset_train[200][0]
# y=NN_model(x)
# print('cnn:',y,'y最大值索引:',torch.argmax(y).tolist(),' real:',dataset_train[200][1])

print(NN_model.parameters())

# 开始测试验证
count_int=0
for data in dataloader_train:
    imgs,targets=data
    # 转到GPU上
    imgs=imgs.cuda()
    targets=targets.cuda()
    outputs=NN_model(imgs)
    if outputs.argmax().tolist()==targets.tolist()[0]:
        count_int=count_int+1

print('验证训练集','totalCount_int:',count_int,'准确率:',count_int/len(dataset_train))


# 测试集
dataset_test=ImageFolder('AntAndBeeClassify2023_03_13\\DataSet\\hymenoptera_data\\val',transform=transform)
print(len(dataset_test))
dataloader_test=DataLoader(dataset_test,batch_size=1,shuffle=True)
# 开始测试验证
count_int=0
for data in dataloader_test:
    imgs,targets=data
    # 转到GPU上
    imgs=imgs.cuda()
    targets=targets.cuda()
    outputs=NN_model(imgs)
    if outputs.argmax().tolist()==targets.tolist()[0]:
        count_int=count_int+1

print('验证测试集','totalCount_int:',count_int,'准确率:',count_int/len(dataset_test))
