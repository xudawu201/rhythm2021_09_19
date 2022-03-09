'''
Author: xudawu
Date: 2021-11-15 14:54:43
LastEditors: xudawu
LastEditTime: 2022-03-03 16:25:55

GRU是Gated Recurrent Unit的缩写，与LSTM最大的不同之处在于GRU将遗忘门和输入门合成一个“更新门”，
同时网络不再额外给出记忆状态，而是将输出结果作为记忆状态不断向后传递，网络的输入和输出都简化。
GRU 对 LSTM 的一些门结构进行了重新设计，归结成两个门结构，一个是 重置门rt（reset gate），另一个是 更新门zt（update gate)
'''
from matplotlib.pyplot import prism
import torch
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim,layers_num,out_features):
        super(GRUModel, self).__init__()#调用Module的构造函数, super(Linear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  #代表GRU层的维度，即每一层GRU层有多少个神经元。
        self.num_layers = layers_num
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        #embedding 嵌入，将矩阵通过乘法降低维度，降低运算量，或者提升维度，把一些其他特征放大
        # torch.nn.Embedding()num_embeddings:一次最大处理多少个样本，embedding_dim:每个样本转为多少维度的tensor即多少列，
        # padding_idx=0:tensor中等于0的变量会处理为全0
        self.embedding = torch.nn.Embedding(num_embeddings=self.input_size,embedding_dim=self.embedding_dim)
        self.GRU = torch.nn.GRU(input_size=self.embedding_dim,# 通过向量嵌入输入，则输入维度与嵌入维度一致
                                hidden_size=self.hidden_size, # 神经元数量
                                num_layers=self.num_layers, # 网络层数
                                batch_first=True, # 第一个维度设为batch,即交换第一二维度参数位置
                                bidirectional=True) # 双向神经网络,即是否考虑前后文
        self.linear=torch.nn.Linear(in_features=self.hidden_size*2,out_features=self.out_features) # 线性变换,从隐藏层到最后的输出层
    def forward(self,x_input):
        x_embedding=self.embedding(x_input) #获取输入并提升维度放大细节

        #每训练一个新batch时初始化一个h0状态,x_embedding.size(0)为每一个训练数据初始化一个h0，size(i),i=0:tesor尺寸的行数,i=1:列数
        h0 = torch.zeros(self.num_layers*2, x_embedding.size(0),self.hidden_size)
        x_GRU, hn = self.GRU(x_embedding, h0) # 返回为每一个神经元的输出
        '''
        output,hn=GRU(input,h0)
        输入:
        input: [seq_len, batch, input_size]
        h0: [num_layers* num_directions, batch, hidden_size]
        输出：
        output: [seq_len, batch, num_directions * hidden_size]
        hn: [num_layers * num_directions, batch, hidden_size]
        '''
        output = self.linear(x_GRU[:, -1, :])# 全连接层,返回最后一个输出
        # output = self.linear(x_GRU)# 全连接层,返回最后一个输出

        return output

#构建数据集
def getDataset(input_tensor, labels_tensor, batch_size):
    from torch.utils import data
    #包装数据存入数据集
    dataset = data.TensorDataset(input_tensor, labels_tensor)
    #从数据集分批次获取一些数据
    dataset_loader = data.DataLoader(dataset, batch_size, shuffle=True)
    return dataset_loader

input_list=[[1,2,2],[1,2,2],[1,2,1],[1,2,1],[1,2,1]]
label_list=[[0,1],[0,1],[1,0],[1,0],[1,0]]
input_tensor = torch.tensor(input_list)
print(input_tensor)
label_tensor = torch.FloatTensor(label_list)
print(label_tensor)
dataset_loader = getDataset(input_tensor, label_tensor,1)
print(dataset_loader)
#机器学习训练过程中得损失函数和优化函数
#交叉熵:判断预测结果和实际结果的一种度量方法。
#错了多少我会用交叉熵告诉你，怎么做才是对的我会用梯度下降算法告诉你#构建数据集
input_size=3 #每个样本所含有的特征数,注意有个默认的0要+1
hidden_size = 4
embedding_dim = 64
layers_num = 1
out_features = 2
GRU_model = GRUModel(input_size=input_size,
                     hidden_size=hidden_size,
                     embedding_dim=embedding_dim,
                     layers_num=layers_num,
                     out_features=out_features)

lr=0.01# 设置学习率，用梯度下降优化神经网络的参数,值越大拟合越快，但可能只是达到局部最优解
optimizer = torch.optim.Adam(GRU_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss() #交叉熵损失函数
input1_list = [[[1, 2, 2]],[[1,2,1]]]
input1_tensor=torch.tensor(input1_list)
print(input1_tensor[0])
output_tensor = GRU_model(input1_tensor[0])
print(output_tensor)

print('开始训练')
GRU_model.train()
epoch=100
for item in range(epoch):
    for input_batch,target_batch in dataset_loader:
        output_tensor = GRU_model(input_batch)
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