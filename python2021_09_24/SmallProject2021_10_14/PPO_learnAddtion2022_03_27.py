'''
Author: xudawu
Date: 2022-03-27 21:34:07
LastEditors: xudawu
LastEditTime: 2022-03-27 21:58:44
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0005
gamma = 0.98 #动作置信度
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3 #同一份数据训练多少次
T_horizon = 5


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(2, 256) #4个输入,车速,车位置,杆角度,杆角速度
        self.fc_pi = nn.Linear(256, 2) #两个输出,向左还是向右
        self.fc_v = nn.Linear(256, 1) #环境价值
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #策略网络
    def pi(self, x, softmax_dim=0):
        #relu()线性激活函数,小于0的数会被限制为0,缓解过拟合和加快运算速度
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        '''
        在机器学习尤其是深度学习中，softmax是个非常常用而且比较重要的函数，尤其在多分类的场景中使用广泛。
        他把一些输入映射为0-1之间的实数，并且归一化保证和为1，因此多分类的概率之和也刚好为1。
        首先我们简单来看看softmax是什么意思。顾名思义，softmax由两个单词组成，其中一个是max。对于max我们都很熟悉，
        比如有两个变量a,b。如果a>b，则max为a，反之为b。用伪码简单描述一下就是 if a > b return a; else b。
        另外一个单词为soft。max存在的一个问题是什么呢？如果将max看成一个分类问题，就是非黑即白，最后的输出是一个确定的变量。
        更多的时候，我们希望输出的是取到某个分类的概率，或者说，我们希望分值大的那一项被经常取到，
        而分值较小的那一项也有一定的概率偶尔被取到，所以我们就应用到了soft的概念，即最后的输出是每个分类被取到的概率。

        softmax第一步就是将模型的预测结果转化到指数函数上，这样保证了概率的非负性。
        将转化后的结果除以所有转化后结果之和，可以理解为转化后结果占总数的百分比。这样就得到近似的概率。

        1.将预测结果转化为非负数,通过指数函数，将实数输出映射到零到正无穷。
        2.各种预测结果概率之和等于1,将所有结果相加，进行归一化。
        返回类型
        #将张量的每个元素缩放到（0,1）区间且和为1
        #返回与输入维度一样的tensor,dim=1按行优先计算,返回每行相加为1,dim=0按列优先计算,返回每列相加为1
        '''
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    #环境价值网络
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    #存入记忆
    def put_data(self, transition):
        self.data.append(transition)

    # 取出当前所有训练的数据,并清空记忆库
    def make_batch(self):
        #取出当前所有训练的数据
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            #游戏结束done_mask=0,游戏未结束done_mask=1
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        # \换行符
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), \
                                          torch.tensor(a_lst), \
                                          torch.tensor(r_lst), \
                                          torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), \
                                          torch.tensor(prob_a_lst)
        #取出记忆后清空记忆库
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    #训练网络
    def train_net(self):
        #取出记忆,
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            # 获取新环境的得分
            td_target = r + gamma * self.v(s_prime) * done_mask
            # 新环境得分减去旧环境的得分
            delta = td_target - self.v(s)
            #tensor转为array
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            # delta[::-1],得分倒序复制一份
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            #翻转为正序
            advantage_lst.reverse()
            #array转回tensor,advantage理解为当前动作的根据环境反馈的置信度
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            #获得动作选择概率列表
            pi = self.pi(s, softmax_dim=1)
            #获得上一次选择的索引
            pi_a = pi.gather(1, a)
            #torch.exp(a),返回e的a次方,e是自然底数约等于2.718,
            #即exp()是将输入映射到0到正无穷的非负数,获得选择当前动作概率的置信度
            ratio = torch.exp(torch.log(pi_a) -torch.log(prob_a))

            #综合评估这次动作选择
            surr1 = ratio * advantage
            #torch.clamp(input, min, max, out=None) → Tensor
            #将输入input张量每个元素的压缩到区间 (min,max)，并返回结果到一个新张量
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            # smooth_l1_loss()计算误差,在真实值和目标值差距过大时,梯度不至于太大
            # torch.min(tensor1,tensor2)逐一比较tensor1和tensor2,返回相应位置的最小值
            loss =F.smooth_l1_loss(self.v(s), td_target.detach()) -torch.min(surr1, surr2)

            #梯度清零
            self.optimizer.zero_grad()
            #反向传播
            loss.mean().backward()
            #梯度优化
            self.optimizer.step()


def main():
    #初始化PPO网络
    model = PPO()
    #初始化得分
    score = 0.0
    #多少次训练打印结果
    print_interval = 10

    for n_epi in range(50):

        #获得初始环境输入
        s = np.array([1,2])
        #是否游戏结束
        done = False
        while not done:
            #每T_horizon次之后或者游戏结束后训练更新网络
            for t in range(T_horizon):
                #返回概率tensor
                prob = model.pi(torch.from_numpy(s).float())
                # 按照传入的prob中给定的概率，在相应的位置处进行取样，取样返回的是与输出相同维度的整数索引tensor
                #不一定会取到最大值所在的索引,按概率取样
                m = Categorical(prob)
                # 返回最大值所在的索引,获得动作选择
                a = m.sample().item()
                #输入动作,返回动作后环境,回报,是否游戏结束
                # s_prime, r, done, info = env.step(a)
                s_prime=np.array([1,2])
                if a==1:
                    r=1
                else:
                    r=0
                done=True
                #记忆池存入信息,prob[a].item()为当前动作的概率
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                #环境继承
                s = s_prime
                #当前得分
                score += r
                # print('t:',t)
                print('a:',a)
                if done:
                    # print('game over')
                    break
                

            #模型训练
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score / print_interval))
            score = 0.0
        prob = model.pi(torch.from_numpy(np.array([1,2])).float())
        # 按照传入的prob中给定的概率，在相应的位置处进行取样，取样返回的是与输出相同维度的整数索引tensor
        #不一定会取到最大值所在的索引,按概率取样
        m = Categorical(prob)
        # 返回最大值所在的索引,获得动作选择
        a = m.sample().item()
        if a==1:
            print('1+2=3')
        else:
            print('err')    

'''
训练PPO
让PPO学会加法
1+2=3
'''
if __name__ == '__main__':
    main()