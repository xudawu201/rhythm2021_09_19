'''
Author: xudawu
Date: 2024-06-09 17:31:00
LastEditors: xudawu
LastEditTime: 2024-06-19 22:11:54
'''

import tkinter as tk  # 导入Tkinter库用于构建图形用户界面

# 初始化Tkinter主窗口
window = tk.Tk()
window.title("TreasureHunt")  # 设置窗口标题

# 设置地图尺寸和单元格大小
# 地图大小为9X9
MAP_SIZE = 9
# 每个单元格的像素大小
CELL_SIZE = 50
# 创建画布
canvas = tk.Canvas(window, width=MAP_SIZE*CELL_SIZE, height=MAP_SIZE*CELL_SIZE, bg='white')
# 将画布添加到窗口中
canvas.pack()
# 空地
space_char='space_tag'
# 宝藏
treasure_char='treasure_tag'
# 玩家
player_char='player_tag'
# 元素类型列表
elementType_list = [space_char,treasure_char,player_char]
# 颜色字典
color_dict={space_char:'skyblue',treasure_char:'gold',player_char:'red'}
# 函数用于在画布上绘制游戏元素
def draw_element(x, y,color):
    """在指定位置绘制游戏元素，如玩家、宝藏或空白格"""
  
    # 计算矩形的像素坐标
    left_x = x * CELL_SIZE
    top_y = y * CELL_SIZE
    right_x = (x + 1) * CELL_SIZE
    bottom_y = (y + 1) * CELL_SIZE
    
    # 使用canvas绘制填充的矩形，无边框
    canvas.create_rectangle(left_x, top_y, right_x, bottom_y, fill=color, outline='')

# 初始化地图数据结构及玩家位置
# 地图初始化，'space_char'代表空地
map_elements = {}
for x in range(MAP_SIZE):
    for y in range(MAP_SIZE):
        map_elements[(x, y)] = space_char

# 宝藏坐标
treasurePos_tuple=(4,4)
map_elements[treasurePos_tuple] = treasure_char  # 放置宝藏
player_pos = (0, 0)  # 玩家起始位置
map_elements[player_pos] = player_char  # 玩家位置标记

# 绘制初始地图
for pos, elementType_str in map_elements.items():
    x, y = pos
    # 绘制元素
    draw_element(x, y,color_dict.get(elementType_str))

# 初始化游戏状态变量
game_over = False  # 游戏是否结束的标志
victory_message = tk.StringVar()  # 用于存储游戏结束时的胜利或失败信息
steps_taken = 0  # 玩家已经移动的步数
MAX_STEPS = 20  # 玩家允许的最大步数

# 创建显示游戏结果的标签
result_label = tk.Label(window, textvariable=victory_message, font=("Helvetica", 16), pady=10)  # 初始化标签但不显示
# 初始隐藏游戏结束信息
result_label.pack_forget()

# 检查游戏是否结束并显示结果的函数
def check_game_over():
    """
    检查游戏是否结束。如果游戏已经结束，则显示游戏结果。

    该函数不接受任何参数，也不返回任何值。
    它的主要作用是更新用户界面，显示游戏结束的状态。
    """
    # 如果游戏已经结束
    if game_over:
        # 显示游戏结果标签
        result_label.pack()
        # 更新界面以显示变化
        window.update_idletasks()
# 更新地图显示函数
def update_map(new_pos):
    """更新玩家位置的显示"""
    global player_pos
    x, y = new_pos
    draw_element(player_pos[0], player_pos[1], color_dict.get(space_char))  # 将原位置重置为空地
    draw_element(x, y, color_dict.get(player_char))  # 在新位置绘制玩家
    player_pos = new_pos  # 更新玩家位置变量
    window.update_idletasks()  # 更新窗口显示

# 玩家移动函数，包括边界检查和游戏结束条件
def move(direction):
    """根据方向移动玩家，并检查是否到达宝藏或超出步数限制"""
    global player_pos, game_over, steps_taken
    
    # 步数用尽，先判断再加步数避免步数溢出和无效步数增加
    if steps_taken == MAX_STEPS:
        victory_message.set("步数用尽，游戏结束。")
        game_over = True
        check_game_over()
        return
    
    # 步数+1
    steps_taken += 1
    # 边界检测
    x, y = player_pos
    # 隐藏游戏提示信息,用于消除走出地图边界的提示
    result_label.pack_forget()
    if direction == "n" and y > 0: y -= 1
    elif direction == "s" and y < MAP_SIZE - 1: y += 1
    elif direction == "e" and x < MAP_SIZE - 1: x += 1
    elif direction == "w" and x > 0: x -= 1
    else:
        # game_over = True
        victory_message.set("你走出了地图！")
        # 显示游戏结果标签
        result_label.pack()
        # 更新界面以显示变化
        window.update_idletasks()

    # 检查是否找到宝藏
    if (x, y) == treasurePos_tuple:
        # 更新地图显示
        update_map((x, y))
        victory_message.set("恭喜！你找到了宝藏！")
        game_over = True
        check_game_over()
        return
    else:
        update_map((x, y))

# 绑定键盘事件处理器
def on_key_press(event):
    """响应键盘事件，根据按键移动玩家并检查游戏状态"""
    check_game_over()  # 检查游戏是否结束
    if game_over:  # 如果游戏结束，则不响应任何键盘事件
        return
    if event.char in ['w', 's', 'a', 'd']:  # 根据键盘输入的方向移动
        move({'w': 'n', 's': 's', 'a': 'w', 'd': 'e'}[event.char])

# 将键盘事件与处理函数绑定
# '<Key>' 是事件类型，代表任何键盘按键被按下
window.bind('<Key>', on_key_press)

# 启动Tkinter主循环
# window.mainloop()

# 在游戏初始化之后，我们添加一个自动游玩的逻辑
def auto_play():
    global game_over, steps_taken
    while not game_over and steps_taken < MAX_STEPS:
        # 简单的自动导航逻辑，这里仅作为一个演示，实际游戏需要更智能的策略
        # 例如，这里我们可以尝试向宝藏的直线方向移动，但不考虑障碍
        dx, dy = treasurePos_tuple[0] - player_pos[0], treasurePos_tuple[1] - player_pos[1]
        
        # 选择移动方向
        if abs(dx) > abs(dy):
            direction = 'e' if dx > 0 else 'w'
        else:
            direction = 's' if dy > 0 else 'n'
        
        move(direction)  # 执行移动
        window.update_idletasks()  # 更新界面
        window.update()  # 更新并等待一小段时间让界面反应
        import time
        time.sleep(0.5)  # 控制自动播放的速度，可调整
        
    # 游戏结束后做一些清理工作，例如关闭窗口
    # if game_over:
    #     window.destroy()

# 修改游戏启动部分，在启动Tkinter主循环之前调用自动游玩函数
# auto_play()
# 注意：由于auto_play包含了循环，上面的window.mainloop()将不再需要，
# 否则程序会因为同时运行两个循环而出现问题。
# window.mainloop()  # 这一行应当被注释掉或删除


# 重置游戏状态到初始值
def reset_game_state():
    global player_pos, game_over, steps_taken
    player_pos = (0, 0)  # 重置玩家位置
    game_over = False  # 重置游戏结束标志
    steps_taken = 0  # 重置步数计数
    # 清除画布上的所有元素
    canvas.delete("all")
    # 重新绘制初始地图
    for pos, elementType_str in map_elements.items():
        x, y = pos
        draw_element(x, y, color_dict.get(elementType_str))
    # 隐藏游戏结果标签
    result_label.pack_forget()

def play_round(round_number):
    """游玩一轮游戏并打印当前轮次"""
    print(f"开始第{round_number}轮游戏")
    auto_play()  # 自动游玩逻辑
    print(f"第{round_number}轮游戏结束")
    reset_game_state()  # 游戏结束后重置状态

def auto_play_rounds():
    """连续游玩三轮游戏"""
    for round_number in range(1, 4):
        play_round(round_number)
    print("三轮游戏全部结束")
    # 这里可以选择关闭窗口或进行其他收尾操作
    # window.destroy()

# 初始化游戏后，直接调用auto_play_rounds函数
# auto_play_rounds()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        """
        初始化Actor类的实例。

        Actor类用于表示一个执行动作的代理，它接收一定的输入，然后通过神经网络模型产生相应的输出。

        参数:
        input_size (int): 输入层的维度，表示输入特征的数量。
        output_size (int): 输出层的维度，表示输出动作的数量。

        """
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # 定义一个全连接层序列，用于从输入到输出的转换。
        self.fc1 = nn.Linear(self.input_size,1048)
        # 上下左右四个输出
        self.fc2 = nn.Linear(1048,self.output_size)
        
    def forward(self, state_tensor,softmax_dim=0):
        """
        前向传播函数，用于计算给定状态的输出分布。

        参数:
        state (tensor): 输入的状态 tensor。

        返回:
        dist (tensor): 经过全连接层处理后得到的概率分布 tensor。
        """
        # 使用全连接层(self.fc)处理输入状态
        x_tensor=self.fc1(state_tensor)
        # LeakyReLU激活函数处理输入x
        x_tensor=nn.functional.leaky_relu(x_tensor)
        # 通过全连接层fc2得到动作网络的输出
        x_tensor=self.fc2(x_tensor)
        # 使用softmax函数将输出转换为概率分布
        prob = nn.functional.softmax(x_tensor, dim=softmax_dim)
        return prob
    
class Critic(nn.Module):
    def __init__(self, input_size):
        """
        初始化批评网络（Critic）。
        
        这个批评网络使用全连接层（fc）来估计输入值的分数。它接收一个特定输入大小的向量，
        并通过两个线性层和一个ReLU激活函数处理它，最终输出一个单一的分数值。
        
        参数:
        input_size -- 输入向量的大小。这个大小决定了输入层的维度。
        """
        super(Critic, self).__init__()  # 调用父类的初始化方法
        # 定义一个全连接层序列，用于从输入到输出的转换。
        self.input_size = input_size
        # 定义一个全连接层序列，用于从输入到输出的转换。
        self.fc1 = nn.Linear(self.input_size,1048)
        # 上下左右四个输出
        self.fc2 = nn.Linear(1048,1)
        
    def forward(self, state_tensor):
        # 使用全连接层(self.fc)处理输入状态
        x_tensor=self.fc1(state_tensor)
        # LeakyReLU激活函数处理输入x
        x_tensor=nn.functional.leaky_relu(x_tensor)
        # 通过全连接层fc2得到评价网络的输出
        value=self.fc2(x_tensor)
        return value

class PPOAgent:
    def __init__(self, input_size, output_size,device):
        """
        初始化代理代理类。

        这个类包含了演员（actor）和评论家（critic）的网络，以及它们的优化器。
        它还定义了一些超参数，用于控制训练过程，比如学习率、梯度裁剪阈值、价值损失系数和熵系数。

        :param input_size: 输入状态的空间大小，用于定义网络的输入层大小。
        :param output_size: 输出动作的空间大小，用于定义网络的输出层大小。
        """

        # 经验池
        self.data = []
        # 学习率
        self.learning_rate = 0.001
        # gamma折扣因子，用于衡量未来奖励的重要性
        self.gamma         = 0.98
        # 优势裁剪因子
        self.lmbda         = 0.95
        # 优势裁剪参数
        self.eps_clip      = 0.1
        # 一次经验池训练次数参数
        self.K_epoch       = 3
        # 初始化演员网络，用于生成动作
        self.actor = Actor(input_size, output_size)
        # 初始化评论家网络，用于评估动作的好坏
        self.critic = Critic(input_size)
        # 使用Adam优化器优化演员网络，学习率为1e-3
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        # 使用Adam优化器优化评论家网络，学习率为3e-4
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        # 将模型迁移到设备
        self.device = device
        self.actor=self.actor.to(self.device)
        self.critic=self.critic.to(self.device)

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        """
        将存储的转换（transition）数据批量转换为PyTorch张量。
        
        这个方法处理的是一个包含多个转换的数据集，每个转换由状态（s）、动作（a）、奖励（r）、
        下一个状态（s'）、动作概率（prob_a）和是否终止（done）组成。它将这些数据转换为
        PyTorch张量，用于训练神经网络。
        
        Returns:
            tuple: 包含六个元素的元组，分别是状态（s）、动作（a）、奖励（r）、
                   下一个状态（s'）、终止标记（done_mask）和动作概率（prob_a）的张量。
        """
        # 初始化用于存储转换数据的列表
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        
        # 遍历数据集中的每个转换
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            # 将转换中的各个元素添加到对应的列表中
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            
            # 根据done值计算done_mask，并添加到done_lst中
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        # 将列表中的数据转换为PyTorch张量
        # 注意：这里使用了分号（;）来连接多个语句，这是一种常见的Python编码风格
        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        
        # 清空数据集，为下次批量处理做准备
        self.data = []
        
        # 返回转换后的张量
        return s, a, r, s_prime, done_mask, prob_a
    
    def updateModel(self):
        """
        训练网络的过程。
        
        通过使用SARSA算法更新网络的策略和价值函数，以优化目标函数。
        """
        # 从经验回放中生成一个批次的数据
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        # 转到cpu或者gpu训练
        s=s.to(self.device)
        a=a.to(self.device)
        r=r.to(self.device)
        s_prime=s_prime.to(self.device)
        done_mask=done_mask.to(self.device)
        prob_a=prob_a.to(self.device)
        # 对于每个训练回合的经验训练K_epoch次，更新网络参数
        for i in range(self.K_epoch):
            # 计算TD目标，即当前状态的价值估计与当前获得奖励之和，即考虑现在奖励和未来奖励
            # gamma折扣因子，用于衡量未来奖励的重要性
            # done_mask确保在结束状态时不考虑未来的奖励
            td_target = r + self.gamma * self.critic(s_prime) * done_mask
            # 计算状态价值的估计误差
            delta = td_target - self.critic(s)
            # 将Tensor转换为numpy数组以进行后续计算
            # 数据转到cpu进行计算
            delta = delta.cpu().detach().numpy()
            # delta = delta.detach()

            # 初始化优势函数的列表
            advantage_lst = []
            advantage = 0.0
            # 通过逆序遍历delta来计算优势函数
            # for delta_t in delta[::-1]:
            for t in reversed(range(0, len(delta))):
                advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                advantage_lst.append([advantage])
            # 翻转列表，以便其顺序与状态序列一致
            advantage_lst.reverse()
            # 将优势函数列表转换为Tensor
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            # 计算当前策略
            pi = self.actor(s, softmax_dim=1)
            # 从当前策略中获取实际采取的动作的概率
            pi_a = pi.gather(1,a)
            # 计算新旧策略的比例
            # 如果新策略更倾向于某个动作（比例大于1），那么就强化这个动作的学习
            # 在训练中，网络更新参数后某个动作概率被优化大了，则表示鼓励这个动作
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(ln(a)-ln(b))=a/b

            # 计算两个策略梯度损失的最小值
            # surr1和surr2分别是未裁剪和裁剪后的策略比率乘以优势函数
            # 这体现了TRPO中的信任区域思想，通过限制策略更新的幅度来维持学习过程的稳定性。
            # 平衡探索与利用，保证策略更新既不过于激进也不过于保守。
            # 通过最小化surrogate loss（即surr1和surr2的最小值），既能鼓励朝着高优势动作移动（通过优势函数放大更新），
            # 又通过裁剪避免更新幅度过大导致策略不稳定。
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + nn.functional.smooth_l1_loss(self.critic(s) , td_target.detach())

            # 输出当前回合的损失平均值
            print('loss mean:',f'{loss.mean().item():.5f}')

            # # 清除梯度缓存，进行反向传播，更新网络参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # # 用损失平均值优化网络参数
            loss.mean().backward()
            # 更新参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
def main():
    global player_pos, game_over, steps_taken
    # 保存模型路径和名字
    # actor模型路径
    actorModelFilePath_str = 'PPOActor.model'
    criticModelFilePath_str = 'PPOCritic.model'
    # 定义设备，用于指定模型运行的设备，如CPU或GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')	# 使用cpu训练
    device = torch.device('cuda')	# 使用gpu训练
    # 初始化PPOAgent
    ppoAgent_model = PPOAgent(4, 4,device)
    # 加载模型
    # model=torch.load(modelFilePath_str)
    #初始化得分
    score = 0.0
    # 动作字典
    action_dict={0:'n',1:'e',2:'s',3:'w'}
    # 每一轮训练最大步数
    T_horizon=21
    for n_epi in range(150):
        
        # 初始化
        reset_game_state()
        #获得初始环境
        # s = env.reset()
        # s等于玩家和宝藏坐标
        s=player_pos+treasurePos_tuple
        # move(direction)  # 执行移动
        window.update_idletasks()  # 更新界面
        window.update()  # 更新并等待一小段时间让界面反应
        import time
        time.sleep(0.5)  # 控制自动播放的速度，可调整
        # 启动Tkinter主循环
        # window.mainloop()
        #是否游戏结束
        done = False
        check_game_over()
        score=0
        # 步数计数
        steps_int=0
        while not done:
            #每T_horizon次之后或者游戏结束后训练更新网络
            for t in range(T_horizon):
                #返回概率tensor
                prob = ppoAgent_model.actor(torch.tensor(s).float().to(device))
                # 创建概率分布模型,然后利用这个分布进行采样、计算对数概率、熵等操作
                m = Categorical(prob)
                # # 根据策略分布采样得到概率索引值，根据概率采样，并不一定会取到最大概率所在的索引值
                # # 通过概率取样增加随机性可以探索更多可能性
                a = m.sample().item()
                # 放弃随机,通过概率取最大值索引
                # a = torch.argmax(prob).item()
                #输入动作,返回动作后环境,回报,是否游戏结束
                print('action_dict:',action_dict.get(a))
                move(action_dict.get(a))
                # 步数加1
                steps_int+=1
                window.update_idletasks()  # 更新界面
                window.update()  # 更新并等待一小段时间让界面反应
                import time
                time.sleep(0.05)  # 控制自动播放的速度，可调整
                # 判断游戏是否结束
                check_game_over()
                done = game_over
                # s_prime等于玩家和宝藏坐标
                s_prime=player_pos+treasurePos_tuple
                # r等于10减去玩家和宝藏坐标差的绝对值,即玩家和宝藏距离越近,reward越高
                r=10-(abs(s_prime[2]-s_prime[0])+abs(s_prime[3]-s_prime[1]))
                # 放大奖励,使目标更重要
                r=r*10
                # 步数越高奖励越低
                r=r-steps_int*8
                # 如果找到宝藏,奖励1000
                if player_pos == treasurePos_tuple:
                    r=r+1000

                # 将当前的经验数据（状态、动作、奖励等）放入模型的经验回放缓冲区
                print(s, 'a:',a, r, s_prime, prob[a].item(),done)
                ppoAgent_model.put_data((s, a, r, s_prime, prob[a].item(), done))


                #环境继承
                s = s_prime
                #当前得分
                score += r
                print('step:',t,'reward',r)
                if done:
                    print('game over')
                    break

            # 用积累的经验进行模型训练
            ppoAgent_model.updateModel()

        # 存储模型
        # 模型转到cpu
        # ppoAgent_model.actor.to('cpu')
        # ppoAgent_model.critic.to('cpu')
        # # 存储actor模型
        # torch.save(ppoAgent_model.actor,actorModelFilePath_str)
        # # 存储critic模型
        # torch.save(ppoAgent_model.critic,criticModelFilePath_str)
        # # 存储完成后再转到gpu继续训练
        # ppoAgent_model.actor.to(device)
        # ppoAgent_model.critic.to(device)

def test():
    global player_pos, game_over, steps_taken
    # 保存模型路径和名字
    actorModelFilePath_str = 'PPOActor.model'
    # 加载模型
    actorAgent_model=torch.load(actorModelFilePath_str)
    # 动作字典
    action_dict={0:'n',1:'e',2:'s',3:'w'}
    for n_epi in range(150):
        
        # 初始化
        reset_game_state()
        #获得初始环境
        # s = env.reset()
        # s等于玩家和宝藏坐标
        s=player_pos+treasurePos_tuple
        # move(direction)  # 执行移动
        window.update_idletasks()  # 更新界面
        window.update()  # 更新并等待一小段时间让界面反应
        import time
        time.sleep(0.5)  # 控制自动播放的速度，可调整
        # 启动Tkinter主循环
        # window.mainloop()
        #是否游戏结束
        done = False
        check_game_over()
        # 步数计数
        steps_int=0
        while not done:
            #返回概率tensor
            prob = actorAgent_model(torch.tensor(s).float())
            # 放弃随机,通过概率取最大值索引
            a = torch.argmax(prob).item()
            #输入动作,返回动作后环境,回报,是否游戏结束
            print('action_dict:',action_dict.get(a))
            move(action_dict.get(a))
            # 步数加1
            steps_int+=1
            window.update_idletasks()  # 更新界面
            window.update()  # 更新并等待一小段时间让界面反应
            import time
            time.sleep(0.05)  # 控制自动播放的速度，可调整
            # 判断游戏是否结束
            check_game_over()
            done = game_over
            # s_prime等于玩家和宝藏坐标
            s_prime=player_pos+treasurePos_tuple
            # r等于10减去玩家和宝藏坐标差的绝对值,即玩家和宝藏距离越近,reward越高
            r=10-(abs(s_prime[2]-s_prime[0])+abs(s_prime[3]-s_prime[1]))
            # 放大奖励,使目标更重要
            r=r*10
            # 步数越高奖励越低
            r=r-steps_int*8
            # 如果找到宝藏,奖励1000
            if player_pos == treasurePos_tuple:
                r=r+1000

            #环境继承
            s = s_prime

            print('step:',steps_int,'reward',r)
            if done:
                print('game over')
                break

'''
训练PPO
让PPOAgent学会寻宝游戏
关键在于奖励的设置
黄色区块为宝藏，红色区块为agent，离宝藏越近得分越高，走的步数越多得分越低
找到宝藏得到大量得分，离宝藏近得分高些，步数增加小幅度降低得分
'''
if __name__ == '__main__':
    main()
    # test()