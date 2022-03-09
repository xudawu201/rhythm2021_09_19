'''
Author: xudawu
Date: 2021-10-29 19:03:38
LastEditors: xudawu
LastEditTime: 2021-10-31 14:51:07
'''
import torch
import os
#简单RNN学习举例。
# RNN（循环神经网络）是把一个线性层重复使用，适合训练序列型的问题。单词是一个序列，序列的每个元素是字母。序列中的元素可以是任意维度的。实际训练中，
# 可以首先把序列中的元素变为合适的维度，再交给RNN层。
#学习 将hello 转为 lolhh。

dict = ['e', 'h', 'l', 'o']  #字典。有4个字母
x_data = [1, 0, 2, 2, 3]  #输入hello在字典中的引索
#后面要用到把引索转化为高维向量的工具，那个工具要求输入是LongTensor而不是默认的floatTensor
x_data = torch.LongTensor(x_data)
#后面用到的交叉熵损失要求输入是一维longTensor。LongTensor的类型是长整型。
y_data = torch.LongTensor([2, 3, 2, 1,1])


# 上边提到的把引索转为高维向量的工具，可以理解为编码器。引索是标量，编出的码可以是任意维度的矢量，在这个例子中
# 把从字典长度为4（只有4个不同的字母的字典）的字典中取出的5个字母hello构成一个序列，编码为5个（一个样本，这个样本有5个元素，也就是
# batch_size=1,seqlen(序列长度）=5，1*5=5）10维向量（bedding_size=10)。
# 然后通过RNN层，降维为5个8维向量(hidding_size=8)。RNN层的输入形状是样本数*序列长度*元素维度，（本例中是1*5*10）RNN的输入有两个，这里我们关心第一个输出
# 它的形状为样本数*序列长度*输出元素维度（本例中为1*5*8）。
# 然后把RNN层的输出视为（样本数*序列长度）*输出元素维度（本例：5*8）的向量交给全连接层降维为5*4。4是因为这是个多分类问题，输入的每个字母对应哪个分类。
# 这里输出分类类别只有4个。（num_class=4)
# 把得到5*4张量交给交叉熵损失计算预测与目标的损失。（后面的工作就是多分类的工作了）。
class RnnModel(torch.nn.Module):
    def __init__(self, dictionary_size, num_class):
        super(RnnModel, self).__init__()
        self.hidden_size = 8
        self.bedding_size = 10
        self.dictionary_size = dictionary_size
        self.num_class = num_class
        self.embeddinger = torch.nn.Embedding(
            self.dictionary_size, self.bedding_size
        )  #把5个引索转化为5个张量。并继承输入的维度。（本例中继承batch_size*seglen),
        # 输出为batch_size*seglen*bedding_size
        self.rnn = torch.nn.RNN(input_size=self.bedding_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True)
        # 指定batch_fisrt=True,则要求输入维度为batch_size*seglen*input_size,否则，要求输入为seglen*input_size。
        # 指定batch_fisrt=True要求的输入形状更方便与构建数据集。数据集的原始维度就是这样的。
        # batch_fisrt默认为false ，之所以为false，是因为seglen*input_size这样的形状更有利于RNN的实现。
        self.linear = torch.nn.Linear(self.hidden_size,
                                      self.num_class)  #10维降为4维。

    def forward(self, x):  #x 的形状为（样本数，序列长度）
        h0 = torch.zeros(
            1, x.size(0),
            self.hidden_size)  #RNN要有两个输入，x为数据，h为序列中上一个元素计算的结果。由于第一个元素没有
        #上一个元素，所以要指定一个初始值。如果没有先验数据，一般设置为全0。它的形状为num_layer*batch_size*hidden_size。num_layers是什么，本人很懒不想画图，其他博客对这个的解释
        #非常清晰。
        x = self.embeddinger(x)
        x, _ = self.rnn(x, h0)  #x的形状为（样本数，序列长度，每个单元的维度）
        x = x.view(-1, self.hidden_size)  #合并前两个维度，放入全连接层计算。如果需要，计算完之后再拆分。
        x = self.linear(x)
        return x


def saveModel_model(model, optimizer, epoch, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, save_dir)


if __name__ == '__main__':
    filePath1 = r'D:\UpUp_2019_06_25\vscode_2020_01_07\python_2021_09_17\SomeResource2021_10_23\model_save.pt'
    # model=torch.load(filePath1)
    model = RnnModel(4, 4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    x_data = x_data.view(-1, 5)
    y_data = y_data.view(-1)
    for epoch in range(5):
        y_hat = model(x_data)
        _, index = torch.max(y_hat, 1)
        index = index.data.numpy()
        loss = criterion(y_hat, y_data)

        print(epoch, ' loss:', loss.item(), 'guess:',
              ''.join([dict[x] for x in index]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.save(model, filePath1)
