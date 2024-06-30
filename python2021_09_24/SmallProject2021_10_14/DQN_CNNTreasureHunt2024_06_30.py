import tkinter as tk  # 导入Tkinter库用于构建图形用户界面

# 初始化Tkinter主窗口
window = tk.Tk()
window.title("TreasureHunt")  # 设置窗口标题

# 设置地图尺寸和单元格大小
# 地图大小为9X9
MAP_SIZE = 9
# 每个单元格的像素大小
CELL_SIZE = 75
# 窗口坐标xy
windowLeft_int=150
windowTop_int=150
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

import time
from datetime import timedelta
# 获取已使用时间
def getTimeUsed(startTime_time):
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    return timedelta(seconds=int(round(time_dif)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, in_channels,out_channels,padding,linear_input_size,out_features):
        super(Actor,self).__init__()
        
        # 输入通道数 
        self.in_channels = in_channels
        # 第一层卷积输出通道数
        self.out_channels = out_channels
        # 边缘填充0圈数
        self.padding=padding
        # 全连接层输入大小
        self.linear_input_size = linear_input_size
        # 全连接层输出类别数
        self.out_features = out_features
        
        # 卷积层1
        self.conv1 = nn.Sequential(
            # 卷积层,padding：前向计算时在输入特征图周围添加0的圈数
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, stride=2, padding=self.padding),
            # 归一化层
            nn.BatchNorm2d(self.out_channels),
            # 池化层
            nn.MaxPool2d(kernel_size=5, stride=2)
        )

        # 卷积层2
        self.conv2 = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels*2, kernel_size=4, stride=2, padding=self.padding),
            # 归一化层
            nn.BatchNorm2d(self.out_channels*2),
            # 池化层
            nn.MaxPool2d(kernel_size=4, stride=2)
        )

        # 卷积层3
        self.conv3 = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=self.out_channels*4, kernel_size=3, stride=1, padding=self.padding),
            # 归一化层
            nn.BatchNorm2d(self.out_channels*4),
            # 池化层
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        # 展平层
        self.flatten = torch.nn.Flatten()
        # 激活层1
        self.gelu = nn.GELU()
        # 全连接层1
        self.fc1 = nn.Linear(self.linear_input_size, self.out_features)

    def forward(self,x):
        # 多层卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 展平层
        x=self.flatten(x)
        # 激活层
        x = self.gelu(x)
        # 全连接层1
        x = self.fc1(x)
        return x

# DQN结构
import collections
import random

class DQN:
    def __init__(self, buffer_capacity,train_epoch,QNetTrain_model,QNetTarget_model,QNetTrain_optimizer,QNetTarget_optimizer,device='cpu'):
        """
        初始化代理代理类。

        这个类包含了演员（actor）和评论家（critic）的网络，以及它们的优化器。
        它还定义了一些超参数，用于控制训练过程，比如学习率、梯度裁剪阈值、价值损失系数和熵系数。

        :param input_size: 输入状态的空间大小，用于定义网络的输入层大小。
        :param output_size: 输出动作的空间大小，用于定义网络的输出层大小。
        """

        # 经验池
        # 创建一个先进先出的队列，当队列满时，旧的经验将被移除
        self.buffer_capacity=buffer_capacity
        self.buffer_list = collections.deque(maxlen=self.buffer_capacity)

        # gamma折扣因子，用于衡量未来奖励的重要性
        self.gamma         = 0.9
        # 一次经验池训练次数参数
        self.train_epoch   = train_epoch

        # 加载模型
        self.QNetTrain_model = QNetTrain_model
        self.QNetTarget_model = QNetTarget_model
        
        # 给网络添加优化器
        self.QNetTrain_optimizer = QNetTrain_optimizer
        self.QNetTarget_optimizer = QNetTarget_optimizer

        # 将模型迁移到设备
        self.device = device
        self.QNetTrain_model=self.QNetTrain_model.to(self.device)
        self.QNetTarget_model=self.QNetTarget_model.to(self.device)

    # 数据转为tensor
    def dataToTensor(self,batch_data):
        # 新建存储经验容器
        state_buffer_tensor=torch.tensor([],dtype=torch.float)
        action_buffer_list = []
        reward_buffer_list = []
        next_state_buffer_tensor=torch.tensor([],dtype=torch.float)
        done_buffer_list = []
        
        # 将采样的数据转为tensor
        for state_tensor,action_int,reward_float,next_state_tensor,done_bool in batch_data:
            # 将当前状态张量拼接到已保存的状态张量列表中
            state_buffer_tensor = torch.cat((state_buffer_tensor, state_tensor.unsqueeze(0)), dim=0)
            # 将采取的行动添加到行动列表中
            action_buffer_list.append([action_int])
            # 将获得的奖励添加到奖励列表中
            reward_buffer_list.append([reward_float])
            # 将下一个状态张量拼接到已保存的下一个状态张量列表中
            next_state_buffer_tensor = torch.cat((next_state_buffer_tensor, next_state_tensor.unsqueeze(0)), dim=0)
            # 将是否完成的标志添加到完成列表中
            # 根据done值计算done_mask，并添加到done_buffer_list中
            if done_bool == True:
                done_int=0
            else:
                done_int=1
            done_buffer_list.append([done_int])

        # 将data列表转换为PyTorch张量
        # 将动作列表转换为PyTorch张量
        action_tensor = torch.tensor(action_buffer_list)
        # 将奖励列表转换为PyTorch张量
        reward_tensor = torch.tensor(reward_buffer_list)
        # 将结束标志列表（done_mask）转换为浮点型的PyTorch张量，表示每个步骤是否结束
        done_tensor = torch.tensor(done_buffer_list, dtype=torch.float)

        # 返回转换后的张量
        return state_buffer_tensor, action_tensor, reward_tensor,next_state_buffer_tensor,done_tensor
    
   # 存储训练数据
    def saveMemory(self, state_tensor, action_int, reward_float, next_state_tensor,done_bool):
        '''
        参数:
        state_tensor: 当前状态的张量表示,state_tensor.shape=torch.Size([data_size])
        action_int: 采取的行动的整数表示,action_int格式单个整型
        reward_float: 采取行动获得的奖励的浮点数表示,reward_float格式单个浮点型
        next_state_tensor: 下一个状态的张量表示,next_state_tensor.shape=torch.Size([data_size])
        done_bool: 该步是否结束的布尔值表示,done_bool格式单个布尔型
        '''
        
        # 将经验数据以元组形式存入经验池
        self.buffer_list.append((state_tensor, action_int, reward_float, next_state_tensor, done_bool))

    # 采样batch_size行数据
    def sampleMemory(self, batch_size):
        # 采样batch_size行数据
        sample_batch_data = random.sample(self.buffer_list, batch_size)

        # 返回这一批数据
        return sample_batch_data
    
    # 采样动作
    def sampleAction(self,x_tensor,out_features,epsilon):
        '''
        根据当前状态样本动作
        本函数用于在给定状态下，根据actor网络预测的动作分布，以一定概率随机选择动作
        参数:
        x_tensor: 网络输出
        epsilon: 随机选择动作的概率阈值
        返回值:
        选择的动作，可以是随机动作或actor网络预测的最有可能的动作
        '''
        # 使用actor网络根据当前状态预测动作分布
        # 生成一个随机数，用于决定是选择随机动作还是根据预测选择动作
        coin = random.random()
        # 如果随机数小于阈值,在0到out_features之前随机选择一个整数,选择随机动作
        if coin < epsilon:
            return random.randint(0, out_features-1)
        else:
            # 否则，选择预测动作分布中概率最大的动作,即网络选择的动作
            return x_tensor.argmax().item()

    # 清空训练数据
    def clearMemory(self):
        self.buffer_list = collections.deque(maxlen=self.buffer_capacity)
    
    #构建数据集
    def getDataset(self,state_tensor, action_tensor, reward_tensor,next_state_tensor,done_tensor,batch_size):
        from torch.utils import data
        #包装数据存入数据集
        dataset = data.TensorDataset(state_tensor, action_tensor, reward_tensor,next_state_tensor,done_tensor)
        #从数据集分批次获取一些数据,设置打乱为False
        dataset_loader = data.DataLoader(dataset, batch_size, shuffle=False)
        return dataset_loader
    
    # 更新QTargent网络参数
    def updateQNetTarget(self):
        """
        这个函数用于更新目标网络参数，使得目标网络逐渐跟随训练网络的最新参数。
        """
        # 将QNetTrain网络的参数复制给QNetTarget网络
        self.QNetTarget_model.load_state_dict(self.QNetTrain_model.state_dict())

    # 利用经验池数据更新模型
    def train_net(self,sample_size,batch_size,n_batch_size):
        """
        训练网络的过程
        DQN:
        实现经验随机采样回放并训练
        """
        # 设置网络为训练模式
        self.QNetTrain_model.train()
        self.QNetTarget_model.train()
        
        # 从经验池中采样sample_size大小的经验数据进行训练
        sample_batch_data=self.sampleMemory(sample_size)
        # 将这一批数据转为tensor并返回
        state_tensor, action_tensor, reward_tensor,next_state_tensor,done_tensor=self.dataToTensor(sample_batch_data)

        # 训练一次用多少数据
        batch_size=batch_size
        # n个batch size之后再更新网络
        n_batch_size=n_batch_size
        # 构建训练数据集
        dataset_loader=self.getDataset(state_tensor, action_tensor, reward_tensor,next_state_tensor,done_tensor,batch_size)

        # 对于每个训练回合的经验训练train_epoch次，更新网络参数
        for epoch_int in range(self.train_epoch):
            
            # 累计损失用于显示
            accumulation_vision_loss = 0.0
            # 累计训练了多少batch size
            accumulation_step=0
            # 更新网络次数
            update_step=0
            # 遍历经验池
            for state_tensor, action_tensor, reward_tensor, next_state_tensor,done_tensor in dataset_loader:
                # 计数batch_size
                accumulation_step=accumulation_step+1

                # 将这一批数据移动到指定的设备（CPU或GPU）上
                state_tensor=state_tensor.to(self.device)
                action_tensor=action_tensor.to(self.device)
                reward_tensor=reward_tensor.to(self.device)
                next_state_tensor=next_state_tensor.to(self.device)
                done_tensor=done_tensor.to(self.device)
                
                # 获得QTrainNet网络输出,QNet的输出表示选择动作的概率有多大,奖励就有多大
                qOutput_tensor=self.QNetTrain_model(state_tensor)
                # 获得QTrainNet网络选择动作位置索引的输出,代表的含义是选择动作的reward
                qActionReward_tensor = qOutput_tensor.gather(1,action_tensor)

                # 获得QNetTarget网络选择动作的输出,代表的含义是选择期望动作的期望reward
                qMaxActionReward_tensor = self.QNetTarget_model(next_state_tensor).max(1)[0].unsqueeze(1)
                
                # 计算目标reward
                targetReward_tensor=reward_tensor+self.gamma*qMaxActionReward_tensor*done_tensor

                # 计算损失,获得QTrainNet和QNetTarget网络的误差
                loss = torch.nn.functional.smooth_l1_loss(qActionReward_tensor, targetReward_tensor)
                
                # 损失平均
                loss=loss.mean()
                
                # 损失除以n个batch size,用于调整累积梯度最后不变
                loss=loss/n_batch_size

                # 反向传播，计算梯度
                loss.backward()

                # 累加损失用于显示
                accumulation_vision_loss = accumulation_vision_loss+loss.item()

                # 每n个batch_size或者到最后一批数据,更新一次QTrainNet网络
                if accumulation_step % n_batch_size==0 or accumulation_step==len(dataset_loader):
                    # 更新参数
                    self.QNetTrain_optimizer.step()

                    # 清除梯度缓存
                    self.QNetTrain_optimizer.zero_grad()

                    # 计数更新网络次数
                    update_step=update_step+1

                    # 打印训练信息
                    print('train epoch:',epoch_int+1,'/',self.train_epoch,'step:',update_step,'/',int(len(dataset_loader)/n_batch_size),'accumulation loss:{:.6f}'.format(accumulation_vision_loss))

                    # 重置损失
                    accumulation_vision_loss=0


import mss
from PIL import Image
import torchvision

# 获取屏幕指定区域的RGB值
def getRegionRGB_tensor(windowRegion_list):
    # 区域坐标
    monitor = (
        windowRegion_list[0], 
        windowRegion_list[1], 
        windowRegion_list[0]+windowRegion_list[2], 
        windowRegion_list[1]+windowRegion_list[3]
        )

    # 抓取图像
    with mss.mss() as mss_img:
        # 截图
        mss_img = mss_img.grab(monitor)
        # 转为PIL图像
        screenshot_PIL = Image.frombytes("RGB", mss_img.size, mss_img.bgra, "raw", "BGRX")
    # ToTensor()会做两件事
    # 将PIL图像的像素值从0-255范围转换到0-1范围，即进行了归一化。
    # 将图像数据的维度从(H, W, C)（高度、宽度、通道数）调整为PyTorch所需的(C, H, W)顺序，
    # 其中C是通道数，对于RGB图像，C等于3。
    transformsImg=torchvision.transforms.Compose([
        # 转为灰度图
        torchvision.transforms.Grayscale(1),
        # 将图片转换为Tensor
        torchvision.transforms.ToTensor(), 
        ])
    image_tensor = transformsImg(screenshot_PIL)
    return image_tensor

def main():
    import time
    global player_pos, game_over, steps_taken

    MAP_SIZE = 9
    # 每个单元格的像素大小
    CELL_SIZE = 75
    # 窗口坐标xy
    windowLeft_int=150
    windowTop_int=150
    # 屏幕缩放比例
    screenScale=1
    # 获得窗口尺寸
    windowRegion_list=[windowLeft_int,windowTop_int,MAP_SIZE*CELL_SIZE,MAP_SIZE*CELL_SIZE]
    # 乘上缩放比例获得真实尺寸
    windowRegion_list=[
        int(screenScale*windowRegion_list[0]),
        int(screenScale*windowRegion_list[1]),
        int(screenScale*windowRegion_list[2]),
        int(screenScale*windowRegion_list[3])
        ]
    print(windowRegion_list)

    # 保存模型路径和名字
    # actor模型路径
    QNetModelFilePath_str = 'QNet.model'

    # 定义设备，用于指定模型运行的设备，如CPU或GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')	# 使用cpu训练
    device = torch.device('cuda')	# 使用gpu训练
    # 学习率
    learning_rate=0.0001
    # 最大步数限制
    maxStep_int=20
    # 采集数据轮数
    totalEpoch_int=300
    # buffer容量大小
    buffer_capacity=1000
    # 当buffer存储了多少数据后开始训练
    start_train_bufferSize_int=200
    # 更新TargetNet的epoch间隔
    updateQTargetNet_int=3
    # 一次经验池数据训练次数,一批网络输出的数据只能计算一次梯度,所以train_epoch只能为1
    train_epoch=3
    # 采样一批数据大小,前期尽量让一个epoch的数据量大于采样的大小。
    # 后期学习能力强，agent不会导致游戏结束，一个epoch的数据量会很多
    sample_size=60
    # 分批训练大小
    batch_size=4
    # n批数据后更新网络
    n_batch_size=5

    # 定义网络参数
    # 输入图像通道数
    in_channels = 1
    # 第一层卷积输出通道数
    out_channels=32
    # 边缘填充0圈数
    padding=2
    # 全连接层输入大小
    linear_input_size=215168
    # 全连接层输出类别数
    QNetTrain_out_features = 4

    # 实例化cnn网络
    # 初始化用于决策的Q网络
    QNetTrain_model = Actor(in_channels,out_channels,padding,linear_input_size,QNetTrain_out_features)
    # 初始化目标Q网络，用于稳定学习过程
    QNetTarget_model = Actor(in_channels,out_channels,padding,linear_input_size,QNetTrain_out_features)

    # 将训练Q网络的参数复制给目标Q网络
    QNetTarget_model.load_state_dict(QNetTrain_model.state_dict())

    # 加载模型,两个Q初始化为一样的网络
    # QNetTrain_model=torch.load(QNetTrainModelFilePath_str)
    # QNetTarget_model=torch.load(QNetTrainModelFilePath_str)

    # 使用Adam优化器优化网络
    QNetTrain_optimizer = optim.AdamW(QNetTrain_model.parameters(), lr=learning_rate)
    QNetTarget_optimizer = optim.AdamW(QNetTarget_model.parameters(), lr=learning_rate)
    # 初始化DQN
    DQNAgent_model = DQN(
        buffer_capacity,
        train_epoch,
        QNetTrain_model,
        QNetTarget_model,
        QNetTrain_optimizer,
        QNetTarget_optimizer,
        device)
    # 动作字典
    action_dict={0:'n',1:'e',2:'s',3:'w'}
    # 训练可视化
    # import wandb
    # wandb.init(project='ppo_train_treasureHunt', name=time.strftime('%m%d%H%M%S'))
    # wandb.login
    # # 训练日志
    # train_log = {}
    for epoch_int in range(totalEpoch_int):
        
        # 初始化
        reset_game_state()

        # 获得初始环境
        # 玩家和宝藏坐标
        state_tuple=player_pos+treasurePos_tuple
        img_tensor=getRegionRGB_tensor(windowRegion_list)
        sInputCPU_tensor=img_tensor.view(
            1,
            img_tensor.shape[0],
            img_tensor.shape[1],
            img_tensor.shape[2])
        # 数据转到device
        sInputDevice_tensor=sInputCPU_tensor.to(device)

        # 更新界面
        window.update_idletasks()
        window.update()  # 更新并等待一小段时间让界面反应
        time.sleep(0.5)  # 控制自动播放的速度，可调整

        #是否游戏结束
        done = False
        # 总得分
        totalReward_float=0.0
        # epoch日志
        # train_log['train/epoch'] = epoch_int
        # 游戏步数
        step_int=0
        # 线性衰减epsilon值，平衡探索与利用网络
        epsilon = max(0.01, 0.08 - 0.01*(epoch_int/200))  # 从8%线性衰减至1%
        # 每maxStep_int次之后或者游戏结束后训练更新网络
        while not done:
            #返回概率tensor
            x_tensor = DQNAgent_model.QNetTrain_model(sInputDevice_tensor)
            # 采用随机选择动作的方式平衡探索和使用网络决策
            # 根据当前状态和epsilon选择动作
            action_int = DQNAgent_model.sampleAction(x_tensor,QNetTrain_out_features,epsilon)

            #输入动作,返回动作后环境,回报,是否游戏结束
            move(action_dict.get(action_int))

            # 更新界面
            window.update_idletasks()
            window.update()  # 更新并等待一小段时间让界面反应
            time.sleep(0.01)  # 控制自动播放的速度，可调整

            # 步数+1
            step_int=step_int+1
            # 判断游戏是否结束
            # 如果步数达到上限,设置done标志
            if step_int==maxStep_int:
                done=True
            else:
                done=False

            # 获取执行动作后的环境
            nextImg_tensor=getRegionRGB_tensor(windowRegion_list)
            sNextInput_tensor=nextImg_tensor.view(
                1,
                nextImg_tensor.shape[0],
                nextImg_tensor.shape[1],
                nextImg_tensor.shape[2])
            
            # nextState等于玩家和宝藏坐标
            nextState_tuple=player_pos+treasurePos_tuple

            # 判断是接近还是远离宝藏
            # 之前距离
            oldDistance_float = abs(state_tuple[0]-state_tuple[2])+abs(state_tuple[1]-state_tuple[3])
            nextDistance_float = abs(nextState_tuple[0]-nextState_tuple[2])+abs(nextState_tuple[1]-nextState_tuple[3])
            if nextDistance_float<oldDistance_float:
                isNearFlag_int=1
            else:
                isNearFlag_int=-1
            # 奖励权重值,用于修正奖励大小
            rewardWeight_float=10.0
            reward_float=rewardWeight_float*isNearFlag_int
            # 步数惩罚因子,用于修正奖励大小
            # 为了避免来回刷奖励,步数惩罚x2需要大于距离奖励
            stepPunishment_float=6
            reward_float=reward_float-stepPunishment_float
            # 如果找到宝藏,奖励
            if player_pos == treasurePos_tuple:
                reward_float=reward_float+300
            
            # 当前总得分
            totalReward_float=totalReward_float+reward_float
            # 将当前的经验数据（状态、动作、奖励等）放入模型的经验回放缓冲区
            stateSave_tensor=sInputCPU_tensor[0]
            sNextSave_tensor=sNextInput_tensor[0]
            DQNAgent_model.saveMemory(stateSave_tensor, action_int, reward_float, sNextSave_tensor, done)

            # 环境继承用于训练
            sInputCPU_tensor = sNextInput_tensor
            # 数据转到device
            sInputDevice_tensor = sInputCPU_tensor.to(device)
            # 位置继承用于计算奖励
            state_tuple=nextState_tuple

            #打印当前步信息
            # 使用detach()截断梯度计算,在训练中会再次用网络进行输出,这里不需要再计算梯度
            prob=torch.nn.functional.softmax(x_tensor,dim=1).detach()
            print('step:',step_int,'action:',action_dict.get(action_int),'reward:',reward_float,'epsilon:',epsilon,'prob:',prob[0][action_int].item(),done)

            if player_pos==treasurePos_tuple or done == True:   
                # print('game over')
                break
                
            # step日志
            # train_log['train/step'] = step_int
            # # 奖励日志
            # train_log['train/reward'] = reward_float
        
        # 每个epoch显示得分
        print('epoch:',epoch_int+1,'/',totalEpoch_int,'totalReward',totalReward_float)
               
        # 总奖励日志
        # train_log['train/totalReward'] = totalReward_float/step_int
        # # wandb写入日志
        # wandb.log(train_log)

        # 当存储经验达到目标条数后开始训练
        if len(DQNAgent_model.buffer_list)>start_train_bufferSize_int:
            print('开始训练')
            # 用积累的经验进行模型训练
            DQNAgent_model.train_net(sample_size,batch_size,n_batch_size)
        
        # 迭代几个epoch后更新目标网络参数
        if (epoch_int+1)%updateQTargetNet_int==0:
            print('更新目标网络参数')
            DQNAgent_model.updateQNetTarget()

        # 存储模型
        # 模型转到cpu
        # DQNAgent_model.QNetTarget_model=DQNAgent_model.QNetTarget_model.to('cpu')
        # # 存储actor模型,将target网络存储到train网络的路径,下次加载需要两个Q网络是一样的，只需要存储一个网络
        # torch.save(DQNAgent_model.QNetTarget_model,QNetTrainModelFilePath_str)
        # # 存储完成后再转到gpu继续训练
        # DQNAgent_model.QNetTarget_model=DQNAgent_model.QNetTarget_model.to(device)

    # 结束日志记录,并上传
    # wandb.finish()


'''
训练PPO
让PPOAgent学会寻宝游戏
关键在于奖励的设置
黄色区块为宝藏，红色区块为agent，离宝藏越近得分越高，走的步数越多得分越低
找到宝藏得到大量得分，离宝藏近得分高些，步数增加小幅度降低得分
'''
if __name__ == '__main__':
    main()
