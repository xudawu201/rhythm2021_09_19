'''
Author: xudawu
Date: 2022-03-13 22:04:04
LastEditors: xudawu
LastEditTime: 2022-03-14 09:56:42
'''
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random

#Hyperparameters
lr_pi = 0.0005 #p网络学习率
lr_q = 0.001 #Q网络学习率
init_alpha = 0.01
gamma = 0.98
batch_size = 32
buffer_limit = 500
tau = 0.01  # for target network soft update
target_entropy = -1.0  # for automated alpha update
lr_alpha = 0.001  # for automated alpha update


class ReplayBuffer():
    def __init__(self):
        #初始化deque长度
        self.buffer = collections.deque(maxlen=buffer_limit)  # deque是为了高效实现插入和删除操作的双向列表,可以在头部插入和删除

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128) #当前环境 三种输入
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))  #以e为底取init_alpha的对数,减少计算量
        self.log_alpha.requires_grad = True  #requires_grad,该方法能够决定自动梯度机制是否需要为当前这个张量计算记录运算操作.
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha) #优化,使动作不被过估计

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        #选取离散正态分布
        dist = Normal(mu, std)#Normal(mean,std)
        # rsample() 对标准正太分布N(0,1)进行采样，然后输出：mean+std×采样值
        action = dist.rsample()
        #最大似然估计
        log_prob = dist.log_prob(action)
        #双曲正切函数的输出范围为(-1，1)，因此将输出映射到-1到1之间。
        real_action = torch.tanh(action)
        #pow(x,y),求x的y次方,等价于x.pow(y),1e-7为 1*10^-7
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) +1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() *(log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64) #三个输入
        self.fc_a = nn.Linear(1, 64) #一个输入,动作只有一种
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))#输入当前环境
        h2 = F.relu(self.fc_a(a))#输入当前动作
        cat = torch.cat([h1, h2], dim=1) #拼接张量,dim=0为按行,dim=1为按列,组成为当前环境和当前动作为输入的张量
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(),self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +param.data * tau)


def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target


def main():
    env = gym.make('Pendulum-v1')
    #初始化记忆库,定义记忆库列表长度
    memory = ReplayBuffer()
    #初始化Q网络
    q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
    #初始化P网络
    pi = PolicyNet(lr_pi)

    #目标网络获得训练网络的参数
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    #得分
    score = 0.0
    #多次打印结果
    print_interval = 20

    for n_epi in range(1000):
        #初始化环境参数
        s = env.reset()
        #未完成游戏
        done = False

        while not done:
            #得到当前动作,和当前动作的最大似然估计
            a, log_prob = pi(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step([2.0 * a.item()])
            memory.put((s, a.item(), r / 10.0, s_prime, done))
            score += r
            s = s_prime
            # env.render()  #显示画面

        if memory.size() > 100:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(
                n_epi, score / print_interval, pi.log_alpha.exp()))
            score = 0.0

    env.close()


'''
使倒立摆直立
'''

if __name__ == '__main__':
    main()