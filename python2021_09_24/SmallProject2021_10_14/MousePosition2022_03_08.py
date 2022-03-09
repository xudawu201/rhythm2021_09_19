'''
Author: xudawu
Date: 2022-03-08 20:42:52
LastEditors: xudawu
LastEditTime: 2022-03-08 20:47:36
'''
from time import sleep
import pyautogui  #鼠标键盘控制
#获取鼠标位置
while True:
    point_point = pyautogui.position()
    print('x:',point_point.x,' y:',point_point.y,end=' ')  # 得到当前鼠标位置；输出：Point(x, y)
    im = pyautogui.screenshot() #返回屏幕的截图，是一个Pillow的image对象
    print(im.getpixel((point_point.x, point_point.y)))
    sleep(1)#停顿1秒