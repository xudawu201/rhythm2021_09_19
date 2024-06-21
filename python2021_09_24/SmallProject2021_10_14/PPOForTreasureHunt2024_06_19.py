'''
Author: xudawu
Date: 2024-06-09 17:31:00
LastEditors: xudawu
LastEditTime: 2024-06-21 12:04:15
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
# 窗口坐标xy
windowLeft_int=100
windowTop_int=100
windowPosition_str=str(MAP_SIZE*CELL_SIZE)+'x'+str(MAP_SIZE*CELL_SIZE)+'+'+str(windowLeft_int)+'+'+str(windowTop_int)
window.geometry(windowPosition_str)
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
        
    def forward(self, state_tensor):
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
        prob = nn.functional.softmax(x_tensor)
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
    def __init__(self, learning_rate,train_epoch,actorModel,criticModel,actor_optimizer,critic_optimizer,device='cpu'):
        """
        初始化代理代理类。

        这个类包含了演员（actor）和评论家（critic）的网络，以及它们的优化器。
        它还定义了一些超参数，用于控制训练过程，比如学习率、梯度裁剪阈值、价值损失系数和熵系数。

        :param input_size: 输入状态的空间大小，用于定义网络的输入层大小。
        :param output_size: 输出动作的空间大小，用于定义网络的输出层大小。
        """

        # 经验池
        self.state_tensor=torch.tensor([],dtype=torch.float)
        self.action_list = []
        self.reward_list = []
        self.next_state_tensor=torch.tensor([],dtype=torch.float)
        self.done_list = []
        self.old_prob_list = []
        # 学习率
        self.learning_rate = learning_rate
        # gamma折扣因子，用于衡量未来奖励的重要性
        self.gamma         = 0.98
        # GAE计算优势时的损失因子
        self.gae_lambda    = 0.95
        # 优势裁剪参数
        self.eps_clip      = 0.1
        # 一次经验池训练次数参数
        self.K_epoch       = train_epoch
        # 加载模型
        self.actor = actorModel
        self.critic = criticModel

        # 给网络添加优化器
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # 将模型迁移到设备
        self.device = device
        self.actor=self.actor.to(self.device)
        self.critic=self.critic.to(self.device)
        # 数据迁移到device
        self.state_tensor=self.state_tensor.to(self.device)
        self.next_state_tensor=self.next_state_tensor.to(self.device)

    # 数据转为tensor
    def dataToTensor(self):
        """
        将存储的转换（transition）数据批量转换为PyTorch张量。
        
        这个方法处理的是一个包含多个转换的数据集，每个转换由状态（s）、动作（a）、奖励（r）、
        下一个状态（s'）、动作概率（prob_a）和是否终止（done）组成。它将这些数据转换为
        PyTorch张量，用于训练神经网络。
        
        Returns:
            return action_tensor, reward_tensor, done_tensor,old_probs_tensor
        """
        # 初始化用于存储转换数据的列表
        tempDone_list = []
        
        # 遍历数据集中的每个转换
        for done in self.done_list:     
            # 根据done值计算done_mask，并添加到done_lst中
            done_int = 0 if done else 1
            tempDone_list.append([done_int])
        
        # 将状态列表转换为浮点型的PyTorch张量
        # 将动作列表转换为PyTorch张量
        action_tensor = torch.tensor(self.action_list)
        # 将奖励列表转换为PyTorch张量
        reward_tensor = torch.tensor(self.reward_list)
        # 将结束标志列表（done_mask）转换为浮点型的PyTorch张量，表示每个步骤是否结束
        done_tensor = torch.tensor(tempDone_list, dtype=torch.float)
        # 将动作概率列表转换为PyTorch张量
        old_prob_tensor = torch.tensor(self.old_prob_list)
        
        # 返回转换后的张量
        return action_tensor, reward_tensor, done_tensor,old_prob_tensor
    
    # 存储训练数据
    def saveMemory(self, state_tensor, action_int, reward_float, next_state_tensor, done_bool, old_prob_float):
        self.state_tensor = torch.cat((self.state_tensor, state_tensor.unsqueeze(0)), dim=0)
        self.action_list.append([action_int])
        self.reward_list.append([reward_float])
        self.next_state_tensor = torch.cat((self.next_state_tensor, next_state_tensor.unsqueeze(0)), dim=0)
        self.done_list.append([done_bool])
        self.old_prob_list.append([old_prob_float])

    # 清空训练数据
    def clearMemory(self):
        self.state_tensor=torch.tensor([],dtype=torch.float)
        self.action_list = []
        self.reward_list = []
        self.next_state_tensor=torch.tensor([],dtype=torch.float)
        self.done_list = []
        self.old_prob_list = []
        # 将数据迁移到设备
        self.state_tensor=self.state_tensor.to(self.device)
        self.next_state_tensor=self.next_state_tensor.to(self.device)
    
    # 计算GAE优势
    def get_GAE_advantage(selft,gamma, gae_lambda, td_delta):
        td_delta = td_delta
        advantage_list = []
        advantage = 0.0
        # 优势值损耗越大,agent越重视未来,但也越难收敛
        for t in reversed(range(0, len(td_delta))):
            advantage = gamma * gae_lambda * advantage + td_delta[t][0]
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    # 利用经验池数据更新模型
    def updateModel(self):
        """
        训练网络的过程。
        
        通过使用SARSA算法更新网络的策略和价值函数，以优化目标函数。
        """
        # 从经验回放中获取数据并转换为tensor
        state_tensor=self.state_tensor
        next_state_tensor=self.next_state_tensor
        action_tensor, reward_tensor, done_tensor,old_prob_tensor = self.dataToTensor()
     
        # 转到cpu或者gpu训练
        state_tensor=state_tensor.to(self.device)
        action_tensor=action_tensor.to(self.device)
        reward_tensor=reward_tensor.to(self.device)
        next_state_tensor=next_state_tensor.to(self.device)
        done_tensor=done_tensor.to(self.device)
        old_prob_tensor=old_prob_tensor.to(self.device)

        # 对于每个训练回合的经验训练K_epoch次，更新网络参数
        for i in range(self.K_epoch):

            # 计算价值估计V(s)
            curValue_tensor = self.critic(state_tensor)

            # 计算下一个状态的价值估计V(s')
            nextValue_tensor = self.critic(next_state_tensor)
            # 计算回报,gamma折扣因子,用于衡量未来奖励的重要性
            returnValue_tensor = reward_tensor + self.gamma * nextValue_tensor * done_tensor
            # 计算状态价值的估计误差
            delta = returnValue_tensor - curValue_tensor
            advantage_tensor=self.get_GAE_advantage(self.gamma,self.gae_lambda,delta)
            # 将优势函数转到device
            advantage_tensor = advantage_tensor.to(self.device)

            # 计算当前策略
            new_prob_tensor = self.actor(state_tensor)
            # 从当前策略中获取实际采取的动作的概率
            new_action_prob_tensor = new_prob_tensor.gather(1,action_tensor)
            # 计算新旧策略的比例
            # 如果新策略更倾向于某个动作（比例大于1），那么就强化这个动作的学习
            # 在训练中，网络更新参数后某个动作概率被优化大了，则表示鼓励这个动作
            ratio = new_action_prob_tensor/old_prob_tensor
            # 计算两个策略梯度损失的最小值
            # surr1和surr2分别是未裁剪和裁剪后的策略比率乘以优势函数
            # 这体现了TRPO中的信任区域思想，通过限制策略更新的幅度来维持学习过程的稳定性。
            # 平衡探索与利用，保证策略更新既不过于激进也不过于保守。
            # 通过最小化surrogate loss（即surr1和surr2的最小值），既能鼓励朝着高优势动作移动（通过优势函数放大更新），
            # 又通过裁剪避免更新幅度过大导致策略不稳定。
            weighted_probs = ratio * advantage_tensor
            weighted_clipped_probs = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage_tensor
            # actor损失
            actor_loss  = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            # critic损失
            critic_loss = nn.functional.mse_loss(curValue_tensor, returnValue_tensor)
            # 总损失
            loss = actor_loss + critic_loss

            # 输出当前回合的损失平均值
            print('loss mean:',f'{loss.mean().item():.5f}')

            # # 清除梯度缓存，进行反向传播，更新网络参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播，计算当前梯度
            loss.mean().backward()
            # 梯度裁剪，norm_type=2: 默认是2范数（即欧几里得范数）
            # 对模型中的所有参数的梯度计算其2范数，如果该范数大于1，则将所有梯度按比例缩小，
            # 使得整体梯度的2范数刚好为1，以此来避免梯度爆炸，稳定训练过程。
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1, norm_type=2)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            # 查看权重梯度值
            # print(self.actor.fc1.weight.grad)
            # print(self.critic.fc1.weight.grad)
            # 更新参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        # 训练完成,清空经验池以存储下一批经验
        self.clearMemory()
def main():
    import time
    global player_pos, game_over, steps_taken
    # 保存模型路径和名字
    # actor模型路径
    actorModelFilePath_str = 'PPOActor.model'
    criticModelFilePath_str = 'PPOCritic.model'
    # 定义设备，用于指定模型运行的设备，如CPU或GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')	# 使用cpu训练
    device = torch.device('cuda')	# 使用gpu训练
    # 学习率
    learning_rate=0.0001
    # 一次经验池数据训练次数
    train_epoch=3
    # 输入特征数
    input_size=4
    # actor输出序列个数
    output_size=4

    # 初始化演员网络，用于生成动作
    # 初始化评论家网络，用于评估动作的好坏
    actor_model = Actor(input_size, output_size)
    critic_model = Critic(input_size)
    # 使用Adam优化器优化网络
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=learning_rate)
    # 加载模型
    # actor_model=torch.load(actorModelFilePath_str)
    # critic_model=torch.load(criticModelFilePath_str)
    # 初始化PPOAgent
    ppoAgent_model = PPOAgent(
        learning_rate,
        train_epoch,
        actor_model,
        critic_model,
        actor_optimizer,
        critic_optimizer,
        device)
    # 加载模型
    # model=torch.load(modelFilePath_str)
    # 动作字典
    action_dict={0:'n',1:'e',2:'s',3:'w'}
    # 每一轮训练最大步数,也等于经验池最大长度
    maxStep_int=21
    epoch_int=300
    for n_epi in range(epoch_int):
        
        # 初始化
        reset_game_state()
        #获得初始环境
        # s = env.reset()
        # s等于玩家和宝藏坐标
        state=player_pos+treasurePos_tuple
        state_tensor=torch.tensor(state).float().to(device)
        # move(direction)  # 执行移动
        window.update_idletasks()  # 更新界面
        window.update()  # 更新并等待一小段时间让界面反应
        time.sleep(0.5)  # 控制自动播放的速度，可调整
        # 启动Tkinter主循环
        # window.mainloop()
        #是否游戏结束
        done = False
        check_game_over()
        # 总得分
        totalReward_float=0.0
        # 步数计数
        steps_int=0
        while not done:
            #每T_horizon次之后或者游戏结束后训练更新网络
            for t in range(maxStep_int):
                #返回概率tensor
                prob = ppoAgent_model.actor(state_tensor)
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
                time.sleep(0.01)  # 控制自动播放的速度，可调整
                # 判断游戏是否结束
                check_game_over()
                done = game_over
                # s_prime等于玩家和宝藏坐标
                nextState=player_pos+treasurePos_tuple
                nextState_tensor=torch.tensor(nextState).float().to(device)
                # r等于玩家和宝藏坐标差即玩家和宝藏距离越近,reward越高
                # 奖励权重值,用于修正奖励大小
                rewardWeight_float=10.0
                # r=nextState[0]-nextState[2]+nextState[1]-nextState[3]+rewardbias_float
                # 判断是接近还是远离宝藏
                # 之前距离
                oldDistance_float = abs(state_tensor[0]-state_tensor[2])+abs(state_tensor[1]-state_tensor[3])
                nextDistance_float = abs(nextState_tensor[0]-nextState_tensor[2])+abs(nextState_tensor[1]-nextState_tensor[3])
                if nextDistance_float<oldDistance_float:
                    isNearFlag_int=1
                elif nextDistance_float==oldDistance_float:
                    isNearFlag_int=0
                else:
                    isNearFlag_int=-1
                r=rewardWeight_float*isNearFlag_int
                # 奖励缩放因子,用于修正奖励大小
                rewardScale_float=10
                r=r*rewardScale_float
                # 步数越高奖励越低
                # 步数奖励缩放因子,用于修正奖励大小
                stepRewardScale_float=5
                r=r-steps_int*stepRewardScale_float
                # 如果找到宝藏,奖励100
                if player_pos == treasurePos_tuple:
                    r=r+1000

                # 当前总得分
                totalReward_float=totalReward_float+r
                # 将当前的经验数据（状态、动作、奖励等）放入模型的经验回放缓冲区
                print(state_tensor, 'a:',a, r, nextState_tensor, prob[a].item(),done)
                ppoAgent_model.saveMemory(state_tensor, a, totalReward_float, nextState_tensor, done, prob[a].item())

                #环境继承
                state_tensor = nextState_tensor
                #当前得分
                print('step:',t,'reward',r,'totalReward_float',totalReward_float)
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
