'''
Author: xudawu
Date: 2024-06-09 17:31:00
LastEditors: xudawu
LastEditTime: 2024-06-10 21:07:04
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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 21

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        # 四个坐标输入
        self.fc1   = nn.Linear(4,1048)
        # 上下左右四个输出
        self.fc_pi = nn.Linear(1048,4)
        # 一个评价价值
        self.fc_v  = nn.Linear(1048,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        """
        计算策略网络的输出概率。

        该方法通过一个全连接层（fc1）将输入x映射到一个中间特征空间，然后通过另一个全连接层（fc_pi）映射到动作空间的概率分布。
        使用ReLU激活函数来引入非线性，最后通过softmax函数将输出转换为概率分布。

        参数:
            x (Tensor): 输入的特征向量。
            softmax_dim (int): 指定softmax计算时的维度，默认为0。

        返回:
            Tensor: 经过softmax激活后的概率分布向量。
        """
        # 通过全连接层fc1和ReLU激活函数处理输入x
        x = F.relu(self.fc1(x))
        # 通过全连接层fc_pi得到策略网络的输出
        x = self.fc_pi(x)
        # 使用softmax函数将输出转换为概率分布
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        """
        计算输入x的值函数v。

        这个函数首先通过一个全连接层（fc1）对输入x进行处理，然后应用ReLU激活函数，
        接着，通过另一个全连接层（fc_v）计算最终的输出v。

        参数:
            x: 输入数据，可以是向量或张量。

        返回值:
            v: 经过两层神经网络处理后的输出值。
        """
        # 应用第一个全连接层和ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用第二个全连接层计算值函数v
        v = self.fc_v(x)
        return v
      
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
        
    def train_net(self):
        """
        训练网络的过程。
        
        通过使用SARSA算法更新网络的策略和价值函数，以优化目标函数。
        """
        # 从经验回放中生成一个批次的数据
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # 对于每个训练回合
        for i in range(K_epoch):
            # 计算TD目标，即当前状态的价值估计与下一个状态的价值估计之和
            td_target = r + gamma * self.v(s_prime) * done_mask
            # 计算状态价值的估计误差
            delta = td_target - self.v(s)
            # 将Tensor转换为numpy数组以进行后续计算
            delta = delta.detach().numpy()

            # 初始化优势函数的列表
            advantage_lst = []
            advantage = 0.0
            # 通过逆序遍历delta来计算优势函数
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            # 翻转列表，以便其顺序与状态序列一致
            advantage_lst.reverse()
            # 将优势函数列表转换为Tensor
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # 计算当前策略
            pi = self.pi(s, softmax_dim=1)
            # 从当前策略中获取实际采取的动作的概率
            pi_a = pi.gather(1,a)
            # 计算新旧策略的比例
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            # 计算两个策略梯度损失的最小值
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            # 输出当前回合的损失
            print('loss:',f'{loss.mean().item():.5f}')

            # 清除梯度缓存，进行反向传播，更新网络参数
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    global player_pos, game_over, steps_taken
    # env = gym.make('CartPole-v1')
    # 保存模型路径和名字
    modelFilePath_str = 'PPONN.model'
    #初始化PPO网络
    model = PPO()
    # 加载模型
    # model=torch.load(modelFilePath_str)
    #初始化得分
    score = 0.0
    #多少次训练打印结果
    print_interval = 20
    # 动作字典
    action_dict={0:'n',1:'e',2:'s',3:'w'}
    sCur, aCur, rCur, sNext, log_probs, doneCur = [], [], [], [],[],[]
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
        # sCur, aCur, rCur, sNext, log_probs, doneCur = [], [], [], [],[],[]
        score=0
        # 步数计数
        steps_int=0
        while not done:
            #每T_horizon次之后或者游戏结束后训练更新网络
            for t in range(T_horizon):
                #返回概率tensor
                prob = model.pi(torch.tensor(s).float())
                # 初始化一个离散分布，用于模拟策略
                m = Categorical(prob)
                # m=prob.detach().numpy()
                # 根据策略分布采样得到动作a,sample随机取
                a = m.sample().item()
                # 放弃随机,通过概率取最大值索引，1代表列方向，即每行都取一个最大值
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
                # s_prime, r, done, info = env.step(a)
                # s_prime等于玩家和宝藏坐标
                s_prime=player_pos+treasurePos_tuple
                # s_prime=torch.tensor([s_prime]).float()
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
                model.put_data((s, a, r, s_prime, prob[a].item(), done))

                # # 将当前状态、动作、log概率和奖励添加到对应的列表中
                # sCur.append(s)
                # aCur.append(a)
                # rCur.append(r)
                # sNext.append(s_prime)
                # log_probs.append(m.log_prob(torch.tensor([a])))
                # doneCur.append(done)

                #环境继承
                s = s_prime
                #当前得分
                score += r
                print('step:',t,'reward',r)
                if done:
                    print('game over')
                    break
                # env.render()  #显示画面

            #模型训练
            # model.train_net(sCur, aCur, rCur, sNext, log_probs, doneCur)
            # 基于积累的经验进行模型训练
            model.train_net()

        # 每隔print_interval个episode，打印一次当前的平均得分
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            # 重置得分
            score = 0.0
        # 存储模型
        torch.save(model,modelFilePath_str)

'''
训练PPO
让PPOAgent学会寻宝游戏
关键在于奖励的设置
黄色区块为宝藏，红色区块为agent，离宝藏越近得分越高，走的步数越多得分越低
找到宝藏得到大量得分，离宝藏近得分高些，步数增加小幅度降低得分
'''
if __name__ == '__main__':
    main()