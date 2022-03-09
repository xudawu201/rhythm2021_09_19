'''
Author: xudawu
Date: 2022-03-03 18:38:56
LastEditors: xudawu
LastEditTime: 2022-03-03 18:39:57
'''
import torch
class LSTM(torch.nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, layers_num,out_features):
        super(LSTM, self).__init__()
        self.input_size = input_size  #所有输入样本有多少种特征
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size  #代表LSTM层的维度，即每一层LSTM层有多少个神经元。
        self.layers_num = layers_num  #神经网络层数
        self.out_features = out_features  #输出多少种类别
        #embedding 嵌入，将矩阵通过乘法降低维度，降低运算量，或者提升维度，把一些其他特征放大
        # torch.nn.Embedding()num_embeddings:一次最大处理多少个样本，embedding_dim:每个样本转为多少维度的tensor即多少列
        self.embedding = torch.nn.Embedding(num_embeddings=self.input_size,embedding_dim=self.embedding_dim)

        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,  # 通过向量嵌入输入，则输入维度与嵌入维度一致
            hidden_size=self.hidden_size,  # 神经元数量
            num_layers=self.layers_num,  # 网络层数
            batch_first=True,  # 第一个维度设为batch,即交换第一二维度参数位置
            bidirectional=True)  # 双向神经网络,即是否考虑前后文)
        self.fc = torch.nn.Linear(in_features=self.hidden_size * 2,
                                  out_features=self.out_features)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(
            x
        )  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layers_num * 2, x.size(0), self.hidden_size)

        c0 = torch.zeros(self.layers_num * 2, x.size(0), self.hidden_size)

        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态,维度与h0和c0一样
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
        out = self.fc(out[:, -1, :])
        return out


#构建数据集
def getDataset(input_tensor, labels_tensor, batch_size):
    from torch.utils import data
    #包装数据存入数据集
    dataset = data.TensorDataset(input_tensor, labels_tensor)
    #从数据集分批次获取一些数据
    dataset_loader = data.DataLoader(dataset, batch_size, shuffle=True)
    return dataset_loader


#生成数据集
input_list = []
label_list = []
for i in range(10):
    for j in range(10):
        # print(j,' ',end='')
        b = []
        b.append(i)
        b.append(j)
        label_list.append(i + j)
        print(b, ' ', end='')
        input_list.append(b)
    print()
print('- ' * 20)
print(input_list)
print(label_list)
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import torch
#给分类编号为数字
lalelEncode_list = LabelEncoder().fit_transform(label_list)
print(lalelEncode_list)
#将标签转换为one-hot编码,两个参数分别为待转列表,种类数
labelOneHot_list = to_categorical(lalelEncode_list, num_classes=19)
print(labelOneHot_list)
label_tensor = torch.tensor(labelOneHot_list)
#显示分类标签和编码
tempLabelName_list = list()
for i in range(len(label_list)):
    if label_list[i] not in tempLabelName_list:
        print('类别:', label_list[i], ' ', '编码:', label_tensor.numpy()[i])
        tempLabelName_list.append(label_list[i])

input_tensor = torch.tensor(input_list)
print(input_tensor)
# print(label_tensor)
#构建数据集
dataset_loader = getDataset(input_tensor, label_tensor, 1)
print(dataset_loader)
#机器学习训练过程中得损失函数和优化函数
#交叉熵:判断预测结果和实际结果的一种度量方法。
#错了多少我会用交叉熵告诉你，怎么做才是对的我会用梯度下降算法告诉你
input_size = 10  #每个样本所含有的特征数,默认有0
hidden_size = 5
embedding_dim = 64
layers_num = 1
out_features = 19
LSTM_model = LSTM(input_size=input_size,
                  embedding_dim=embedding_dim,
                  hidden_size=hidden_size,
                  layers_num=layers_num,
                  out_features=out_features)

lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()  #交叉熵损失函数
input1_list = [[input_list[0]]]
print('input1_list:', input1_list)
input1_tensor = torch.tensor(input1_list)
print('input1_tensor:', input1_tensor[0])
output_tensor = LSTM_model(input1_tensor[0])
print(output_tensor)

import time
from datetime import timedelta


# 获取已使用时间
def getTimeUsed(startTime_time):
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    return timedelta(seconds=int(round(time_dif)))


starTime_time = time.time()

print('开始训练')
LSTM_model.train()
epoch = 300
for item in range(epoch):
    for input_batch, target_batch in dataset_loader:
        output_tensor = LSTM_model(input_batch)
        # print('output_tensor:',output_tensor.detach().numpy())
        loss = criterion(output_tensor, target_batch)
        print('loss:{:.4f}'.format(loss.item()))

        # print(output_tensor)
        # print(target_batch)

        #梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #梯度优化
        optimizer.step()
#生成验证集
# test_list=[[[1,2]],[[3,5]],[[1,8]],[[5,8]],[[2,9]],[[5,7]],[[2,4]],[[2,5]],[[1,5]],[[3,6]],[[6,9]],[[3,1]],[[7,2]],[[7,3]],[[5,2]]]
import random
import numpy

random = numpy.random.RandomState(0)
test_list = []
for item in range(20):
    temp1_list = []
    a = random.randint(0, 9)
    temp1_list.append(a)
    a = random.randint(0, 9)
    temp1_list.append(a)
    temp2_list = []
    temp2_list.append(temp1_list)
    test_list.append(temp2_list)
# print(test_list)
test_tensor = torch.tensor(test_list)
import numpy as np
for i in range(len(test_list)):
    #获得网络模型输出tensor
    pre_tensor = LSTM_model(test_tensor[i])
    # print(pre_tensor)
    #获得网络模型输出并根据输出分类
    pre_narry = pre_tensor.detach().numpy()
    print('当前计算结果为:')
    print(test_list[i][0], '=', np.argmax(pre_narry))

print('本次训练+验证用时:', getTimeUsed(starTime_time))