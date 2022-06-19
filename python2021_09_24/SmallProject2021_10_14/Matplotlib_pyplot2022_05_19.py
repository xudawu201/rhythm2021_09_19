'''
Author: xudawu
Date: 2022-05-19 14:48:13
LastEditors: xudawu
LastEditTime: 2022-05-19 16:55:17
'''
# 绘图包
from random import randint
import matplotlib.pyplot as plt
# 1.静态图
plt.figure()#建立画板
x_list = [1,2,3,4,5] # x轴数据
y_list = [4,5,6,7,8] #y轴数据
'''
plt.plot(x, y, ls='-', lw=2, label='xxx', color='g' )
x: x轴上的值
y: y轴上的值
ls:线条风格 (linestyle)
lw:线条宽度 (linewidth)
label:标签文本
'''
# plot画线,默认颜色为blue,
# plot函数画出一系列的点，并且用线将它们连接起来
plt.plot(x_list, y_list, linewidth=2,label='line')
plt.legend()  #打上标签
# 设置图表标题，并给坐标轴添加标签
plt.title('line01',fontsize=20)
plt.xlabel('x01', fontsize=12)
plt.ylabel('y01', fontsize=12)

# plt.show()
# 2.动态图
# matplotlib的显示模式默认为阻塞（block）模式（即：在plt.show()之后，程序会暂停到那儿，并不会继续执行下去）。
plt.ion()# 打开交互模式
# 在plt.show()之前加plt.ioff()，如果不加，界面会一闪而过，并不会停留。
# 在交互模式下：
# 1、plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()。
# 2、如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，
# 则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。
#实时绘制折线图
xDynamic_list=[]
yDynamic_list=[]
def dynamicDraw(x,y):
    plt.cla() # 清除之前的绘图
    xDynamic_list.append(x)  # 添加x轴数据
    yDynamic_list.append(y)  # 添加y轴数据
    plt.plot(xDynamic_list, yDynamic_list,color='b',ls='-',label='line')
    plt.pause(0.1)#暂停0.1秒查看图像

# 设置图表标题，并给坐标轴添加标签
plt.title('line',fontsize=20)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
for i in range(1000):
    x = i
    # y = randint(0,100)
    y=i*i
    dynamicDraw(x,y)
