'''
Author: xudawu
Date: 2022-03-06 21:22:02
LastEditors: xudawu
LastEditTime: 2022-03-08 23:12:10
'''
import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, image_high,image_width,in_channels, out_channels,kernel_size, stride, out_features):
        super(CNN,self).__init__()  #调用Module的构造函数, super(Linear, self).__init__()
        self.image_high = image_high
        self.image_width = image_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_features = out_features
        self.padding = int((self.kernel_size - stride) / 2)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,  #输入图像通道数
                out_channels=self.out_channels,  #卷积产生的通道数,即输出的深度,多少个filter
                kernel_size=self.kernel_size,  #卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3卷积核
                stride=self.stride,  #扫描两个相邻区域之间的步长
                padding=self.padding ), #使卷积后图片尺寸不变化,卷积层1
            nn.ReLU(),  #激活函数
            nn.MaxPool2d(kernel_size=self.kernel_size), #池化层,降维，减少计算开销。stride默认值是kernel_size
            nn.Conv2d(self.out_channels,
                      self.out_channels * 2,
                      self.kernel_size,
                      self.stride,
                      padding=self.padding),  #卷积层2
            nn.ReLU(),  #激活函数
            nn.MaxPool2d(kernel_size=self.kernel_size),
        )
        self.out = nn.Linear(in_features=153360, out_features=out_features)

    #三维数据展平成2维数据
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 将四维张量转换为二维张量之后，才能作为全连接层的输入
        output = self.out(x)
        return output


#定义网络参数
image_high = 1080
image_width=1920
in_channels = 3
out_channels = 3
kernel_size = 3
stride = 1
out_features = 2
CNN_model = CNN(image_high=image_high,
                image_width=image_width,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                out_features=out_features)
print(CNN_model)  #打印结构
import time
from datetime import timedelta


# 获取已使用时间
def getTimeUsed(startTime_time):
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    return timedelta(seconds=int(round(time_dif)))


import pyautogui
import numpy
import torch
gameover=[252,62,57]
block1=[37,710]
block2 = [210, 710]
block3 = [350, 710]
block4 = [460, 710]
print(pyautogui.size())


#获取当前屏幕的像素值
def getScreenshotPixel():
    img = pyautogui.screenshot()  #获取屏幕
    imgPixel_array=numpy.array(img)#获取当前屏幕的像素值
    return imgPixel_array

#屏幕像素值转换为神经网络输入
def getInputTensor(imgPixel_array):
    input_tensor = torch.FloatTensor(imgPixel_array)
    input_tensor = input_tensor.reshape(1, 3, 1080, 1920)  #转为4维数组
    return input_tensor

#查找屏幕重开标志
def getScreenIfRestart():
    if pyautogui.locateOnScreen(
            r'D:\UpUp_2019_06_25\vscode_2020_01_07\python_2021_09_17\restart.PNG',confidence=0.7):
        return 1
    else :
        return 0
#按下鼠标
def mousePress():
    x,y=pyautogui.position()
    pyautogui.click(x,y)

# imgPixel_array=getScreenshotPixel()
# input_tensor =getInputTensor(imgPixel_array)
# starTime_time = time.time()
# output = CNN_model(input_tensor)
# print('timeUsed:', getTimeUsed(starTime_time))
# print('output:')
# print(output)

#CNN网络训练
lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()  #交叉熵损失函数

#无监督学习建立标签
label_list=[[[1,0]],[[0,1]]] #[1,0]正确，[0,1]错误
label_tensor=torch.FloatTensor(label_list)
print(label_tensor[0])

#训练网络
print('开始训练')
CNN_model.train()
epoch = 10
for item in range(epoch):
    #记录时间戳
    starTime_time = time.time()
    #获取屏幕像素值
    imgPixel_array=getScreenshotPixel()
    #像素值转换为tensor
    input_tensor =getInputTensor(imgPixel_array)
    #输出神经网络得出输出
    output = CNN_model(input_tensor)
    #按下鼠标
    mousePress()
    #判断此时屏幕有无出现重开按钮
    index=getScreenIfRestart()
    #如果游戏失败则设置延时
    if index==1:
        time.sleep(0.5)
    #计算损失
    loss=criterion(output,label_tensor[index])
    #获取时间
    timeUsed=getTimeUsed(starTime_time)
    print('output_tensor:', output.detach().numpy())
    print('loss:{:.4f},timeUsed:{:}'.format(loss.item(), timeUsed))


    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #梯度优化
    optimizer.step()
