'''
Author: xudawu
Date: 2022-03-03 16:13:12
LastEditors: xudawu
LastEditTime: 2022-03-14 11:16:49
'''
import torch
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers_num, out_features):
        super(LSTM, self).__init__()
        self.input_size = input_size  #每个输入样本有多少输入特征
        self.hidden_size = hidden_size  #代表LSTM层的维度，即每一层LSTM层有多少个神经元。
        self.layers_num = layers_num  #神经网络层数
        self.out_features = out_features  #输出多少种类别

        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,  # 输入维度
            hidden_size=self.hidden_size,  # 神经元数量
            num_layers=self.layers_num,  # 网络层数
            batch_first=True,  # 第一个维度设为batch,即交换第一二维度参数位置
            bidirectional=True)  # 双向神经网络,即是否考虑前后文)
        self.fc = torch.nn.Linear(in_features=self.hidden_size * 2,
                                  out_features=self.out_features)

    def forward(self, x):
        # LSTM(input,h,c) input需要是三维向量
        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layers_num * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.layers_num * 2, x.size(0), self.hidden_size)

        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态,维度与h0和c0一样
        '''
        激活函数是用来加入非线性因素的，提高神经网络对模型的表达能力，解决线性模型所不能解决的问题
        relu()线性激活函数,小于0的数会被限制为0,缓解过拟合和加快运算速度
        '''
        x=torch.relu(x)
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


input_list = [[[1, 2, 2]], [[1, 2, 2]], [[1, 2, 1]], [[1, 2, 1]], [[1, 2, 1]]]
label_list = [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]]
input_tensor = torch.FloatTensor(input_list)
print(input_tensor)
label_tensor = torch.FloatTensor(label_list)
print(label_tensor)
dataset_loader = getDataset(input_tensor, label_tensor, 1)
print(dataset_loader)
#机器学习训练过程中得损失函数和优化函数
#交叉熵:判断预测结果和实际结果的一种度量方法。
#错了多少我会用交叉熵告诉你，怎么做才是对的我会用梯度下降算法告诉你#构建数据集
input_size = 3  #每个样本所含有的特征数,注意有个默认的0要+1
hidden_size = 4
layers_num = 1
out_features = 2
LSTM_model = LSTM(input_size=input_size,
                  hidden_size=hidden_size,
                  layers_num=layers_num,
                  out_features=out_features)

lr = 0.01  # 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()  #交叉熵损失函数
input1_list = [[[[1, 2, 2]]], [[[1, 2, 1]]]]
input1_tensor = torch.FloatTensor(input1_list)
print(input1_tensor[0])
output_tensor = LSTM_model(input1_tensor[0])
print(output_tensor)

print('开始训练')
LSTM_model.train()
epoch = 100
for item in range(epoch):
    for input_batch, target_batch in dataset_loader:
        output_tensor = LSTM_model(input_batch)
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
