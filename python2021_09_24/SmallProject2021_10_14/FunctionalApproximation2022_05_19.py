'''
Author: xudawu
Date: 2021-10-14 15:13:15
LastEditors: xudawu
LastEditTime: 2022-05-20 16:31:53
'''
import torch
#定义神经网络
class NeuralNet(torch.nn.Module):
    def __init__(self,in_features,neureCell_numbers,out_features):
        super(NeuralNet, self).__init__()
        self.in_features=in_features
        self.neureCell_numbers=neureCell_numbers #神经元数量
        self.out_features=out_features
        self.linear1=torch.nn.Linear(in_features=self.in_features,out_features=self.neureCell_numbers)#隐藏层
        self.linear2=torch.nn.Linear(in_features=self.neureCell_numbers,out_features=self.out_features)#输出层
    def forward(self,x):
        x=self.linear1(x)
        x=self.linear2(x)
        return x

# 实例化网络
in_features=1 #输入特征
out_features=1  #输出特征
neureCell_numbers=10 #神经元数量
nn_Net=NeuralNet(in_features=in_features,neureCell_numbers=neureCell_numbers,out_features=out_features)

lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
optimizer = torch.optim.Adam(nn_Net.parameters(), lr=lr)
criterion = torch.nn.MSELoss()  #均方差损失函数

# 数据预处理
import numpy
x=numpy.linspace(-2*numpy.pi,2*numpy.pi,500).tolist()
y=numpy.sin(x).tolist()
# print(x)
# print(y)
x_tensor=torch.FloatTensor(x).view(len(x),-1)
y_tensor=torch.FloatTensor(y).view(len(x),-1)
print(x_tensor.size())
print(y_tensor.size())
out=nn_Net(x_tensor[0])
print(out)

# 训练
print('start train')
epoch=500 # 训练次数
import matplotlib.pyplot as plt
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
        output_tensor=nn_Net(x_tensor[step])
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