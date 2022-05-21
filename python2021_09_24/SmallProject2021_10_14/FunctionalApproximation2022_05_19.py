'''
Author: xudawu
Date: 2021-10-14 15:13:15
LastEditors: xudawu
LastEditTime: 2022-05-21 18:06:27
'''
import torch
import matplotlib.pyplot as plt
import time
from datetime import timedelta

# 获取已使用时间
def getTimeUsed(startTime_time):
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    return timedelta(seconds=int(round(time_dif)))
    
#定义简单神经网络
class SimpleNet(torch.nn.Module):
    def __init__(self,in_features,neureCell_numbers,out_features):
        super(SimpleNet, self).__init__()
        self.in_features=in_features
        self.neureCell_numbers=neureCell_numbers #神经元数量
        self.out_features=out_features
        # 隐藏层
        self.linear1=torch.nn.Linear(in_features=self.in_features,out_features=self.neureCell_numbers)
        # 输出层
        self.linear2=torch.nn.Linear(in_features=self.neureCell_numbers,out_features=self.out_features)
    def forward(self,x):
        # 隐藏层
        x=self.linear1(x)
        # 激活
        # x=torch.relu(x) #不使用激活函数反而拟合效果更好
        # x=torch.sigmoid(x) #在此场景中sigmoid效果要好于relu,但也不如不使用激活函数
        # x=torch.tanh(x) #在此场景中效果类似于sigmoid
        # 输出层
        x=self.linear2(x)
        return x

#定义GRU神经网络
class GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size,num_layers,out_features):
        super(GRU, self).__init__()#调用Module的构造函数, super(Linear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  #代表隐藏层输出特征的个数
        self.num_layers = num_layers    #代表GRU的隐藏层的层数
        self.out_features = out_features

        self.GRU = torch.nn.GRU(input_size=self.input_size,# 输入特征数
                                hidden_size=self.hidden_size, # 隐层的输出特征的个数,也等于输出的特征个数和每个hn的特征个数
                                num_layers=self.num_layers, # 隐藏层堆叠的高度，用于增加隐层的深度。也等于hn的个数
                                batch_first=True, # 第一个维度设为batch,即交换第一二维度参数位置
                                bidirectional=True) # 双向神经网络,即是否考虑前后文
        self.linear=torch.nn.Linear(in_features=self.hidden_size*2,out_features=self.out_features) # 线性变换,最后的输出层
    def forward(self,x_input):
        x, hn = self.GRU(x_input) # 返回GRU输出
        # GRU自带激活函数,不需要激活函数
        # x=torch.relu(x) # 此效果在隐藏层多层时才发挥作用
        # x=torch.sigmoid(x) #此效果比relu略好
        # x=torch.tanh(x) #此效果比sigmoid差
        # 逻辑回归使用sigmoid>relu>tanh
        output = self.linear(x[:, -1, :])# 全连接层,返回最后一个输出
        return output

#构建数据集
def getDataset(input_tensor, labels_tensor, batch_size):
    from torch.utils import data
    #包装数据存入数据集
    dataset = data.TensorDataset(input_tensor, labels_tensor)
    #从数据集分批次获取一些数据
    dataset_loader = data.DataLoader(dataset, batch_size, shuffle=False)
    print('构建数据集成功')
    return dataset_loader

# simple网络数据预处理
def simepleDataProcess(net):
    tempX=x[0]
    print('tempX:',tempX)
    tempX_tensor=torch.FloatTensor(tempX).view(len(tempX),-1) # liner输入需要2维
    print('tempX_tensor.size():',tempX_tensor.size())
    print('tempX_tensor:',tempX_tensor)
    print('- '*20)
    # 使用simple网络
    out=net(tempX_tensor)
    print('out:',out.size())
    print('out:',out)

    x_tensor=torch.FloatTensor(x).view(len(x),-1)
    y_tensor=torch.FloatTensor(y).view(len(x),-1)
    # print(x_tensor.size())
    # print(y_tensor.size())
    return x_tensor,y_tensor

def GRUDataProcess(net):
    tempX=x[0]
    print('tempX:',tempX)
    tempX_tensor=torch.FloatTensor(tempX).view(len(tempX),1,-1) #转为3维,GRU输入需要3维
    print('tempX_tensor.size():',tempX_tensor.size())
    print('tempX_tensor:',tempX_tensor)
    print('- '*20)
    # 使用simple网络
    out=net(tempX_tensor)
    print('out:',out.size())
    print('out:',out)

    # 构建训练数据集
    print('构建训练数据集')
    x_tensor=torch.FloatTensor(x).view(len(x),1,-1)#转为3维
    y_tensor=torch.FloatTensor(y).view(len(x),-1)
    print(x_tensor.size())
    print(y_tensor.size())
    dataset_loader=getDataset(x_tensor,y_tensor,batch_size=1)
    return dataset_loader

# 简单网络训练
def simpleTrain(simpleNet,x_tensor,y_tensor,optimizer,criterion):
    # 训练
    print('start train')
    epoch=50 # 训练次数
    # plt.ion()#开启交互模式
    for i in range(epoch):
        plt.cla() # 清除之前的绘图
        xDynamic_list=[]
        yDynamic_list=[]
        def dynamicDraw(xd,yd):
            xDynamic_list.append(xd)  # 添加x轴数据
            yDynamic_list.append(yd)  # 添加y轴数据
            # ls：线条风格 (linestyle)
            plt.plot(x,y,color='g',ls='-',label='line') # 原函数
            plt.plot(xDynamic_list, yDynamic_list,color='b',ls='-',label='line') # 拟合函数
            plt.pause(0.1)#暂停0.1秒查看图像
        finalx_list=[]
        finaly_list=[]
        for step in range(len(x_tensor)):
            # 获取网络输出
            output_tensor=simpleNet(x_tensor[step])
            # 计算误差
            loss=criterion(output_tensor,y_tensor[step])
            x_list=x[step]
            y_list=output_tensor.tolist()[0]
            # 输出训练信息
            print('totalEpoch:',i+1,'/',epoch,'epoch:',step+1,'/',len(x),'output:',y_list,'loss:',f'{loss.item():.8f}')

            # 反向传播
            # 梯度清零
            optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 绘图
            # dynamicDraw(x_list,y_list)
            finalx_list.append(x_list)
            finaly_list.append(y_list)
        # plt.ioff()#关闭交互
        # plt.plot(x,y,color='g',ls='-',label='realLine') # 原函数
        # plt.plot(finalx_list, finaly_list,color='b',ls='-',label='preLine') # 拟合函数
        # plt.legend(loc='upper right')
        # plt.show()
    plt.ioff()#关闭交互
    plt.plot(x,y,color='g',ls='-',label='realLine') # 原函数
    plt.plot(finalx_list, finaly_list,color='b',ls='-',label='preLine') # 拟合函数
    plt.legend(loc='upper right')
    plt.show()
    
# 动态绘图
def dynamicDraw(realX,realY,x_list,y_list,xd,yd):
    plt.cla() # 清除之前的绘图
    x_list.append(xd)  # 添加x轴数据
    y_list.append(yd)  # 添加y轴数据
    # ls：线条风格 (linestyle)
    plt.plot(realX,realY,color='g',ls='-',label='line') # 原函数
    plt.plot(x_list, y_list,color='b',ls='-',label='line') # 拟合函数
    plt.pause(0.1)#暂停0.1秒查看图像
    return x_list,y_list

import csv
#存储CSV文件,每一行为一个数据
def saveCsvFile(filePath,content_list):
    # 使用open()方法打开这个csv文件,newline=''表示不换行存储,w重写
    fileOpen = open(filePath, mode='w', encoding='utf-8-sig',newline='')
    # 写入csv文件
    fileOpenCsv=csv.writer(fileOpen)
    # writerows()将一个二维列表中的每一个列表写为一行。
    fileOpenCsv.writerows(content_list)
    # 关闭文件
    fileOpen.close()
    print('save file success')

# 神经网络训练
def NNTarin(nNet,epoch,dataset_loader,optimizer,criterion):
    loss_list=[]
    for i in range(epoch):
        step=0 # 训练步数
        x_list=[] # 训练x轴历史数据
        y_list=[] # 训练y轴历史数据
        # 每轮最终训练数据
        finalx_list=[]
        finaly_list=[]
        plt.ion() # 开启交互模式
        temploss=0
        for x_tensor,y_tensor in dataset_loader:
            startTime=time.time()
            step+=1 
            # 获取网络输出
            outPut_tensor=nNet(x_tensor)
            # 计算损失
            loss=criterion(outPut_tensor,y_tensor)
            # 反向传播
            # 梯度清零
            optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 训练x和y信息
            xTrain=x_tensor.tolist()[0][0]
            yTrain=outPut_tensor.tolist()[0][0]
            x_list.append(xTrain)
            y_list.append(yTrain)

            timeUsed=getTimeUsed(time.time())
            # 输出训练信息
            print('epoch:',i+1,'/',epoch,'step:',step,'/',len(dataset_loader),'loss:',f'{loss.item():.8f}','timeUsed:',timeUsed)
            # 绘图
            # dynamicDraw(x,y,x_list,y_list,xTrain,yTrain)
            finalx_list.append(xTrain)
            finaly_list.append(yTrain)
            temploss+=loss.item()
        # plt.ioff() #交互关闭
        # plt.plot(x,y,color='g',ls='-',label='realLine') # 原函数
        # plt.plot(finalx_list, finaly_list,color='b',ls='-',label='preLine') # 拟合函数
        # plt.show() 
        loss_list.append([temploss])
    plt.ioff() #交互关闭
    plt.plot(x,y,color='g',ls='-',label='realLine') # 原函数
    plt.plot(finalx_list, finaly_list,color='b',ls='-',label='preLine') # 拟合函数
    plt.show()
    saveCsvFile('loss.csv',loss_list)

# 数据集获取
import numpy
totalNumber=500
x=numpy.linspace(-2*numpy.pi,2*numpy.pi,totalNumber).reshape(totalNumber,-1).tolist()
y=numpy.sin(x).reshape(totalNumber,-1).tolist()
# print(x[0])
# print(y[0])

# 构建simpleNet
def builtSimpleNet():
    # 实例化simpleNet网络
    in_features=1 #输入特征
    out_features=1  #输出特征
    neureCell_numbers=10 #隐藏层神经元数量
    nn_Net=SimpleNet(in_features=in_features,neureCell_numbers=neureCell_numbers,out_features=out_features)

    lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
    optimizer = torch.optim.Adam(nn_Net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()  #均方差损失函数
    return nn_Net,optimizer,criterion

# 构建GRU
def builtGRUNet():
    # 实例化GRU网络
    input_size = 1  # 输入特征数
    hidden_size = 10  # 隐藏层输出特征的个数
    num_layers = 1    #代表GRU的隐藏层的层数
    out_features = 1 #最后全连接层输出特征
    GRUNet = GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                out_features=out_features)
    lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
    # 定义优化器
    optimizer = torch.optim.Adam(GRUNet.parameters(), lr=lr)
    # 定义损失函数
    criterion = torch.nn.MSELoss()  #均方差损失函数
    return GRUNet,optimizer,criterion

# # 构建simpel网络
# nn_Net,optimizer,criterion=builtSimpleNet()
# # 获取数据
# x_tensor,y_tensor=simepleDataProcess(nn_Net)
# # # 开始训练
# simpleTrain(nn_Net,x_tensor,y_tensor,optimizer,criterion)

# 构建GRU网络
GRUNet,optimizer,criterion=builtGRUNet()
# 获取数据
dataset_loader=GRUDataProcess(GRUNet)
# 开始训练
epoch=5
NNTarin(GRUNet,epoch,dataset_loader,optimizer,criterion)
