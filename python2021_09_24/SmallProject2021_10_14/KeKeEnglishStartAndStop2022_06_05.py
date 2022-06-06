'''
Author: xudawu
Date: 2022-06-05 19:55:20
LastEditors: xudawu
LastEditTime: 2022-06-06 19:49:52
'''
# 实现可可英语的启动和停止功能
import pyautogui #查找ui并自动化
import keyboard #监听键盘

# 文件路径
startImagePath='start.png'
stopImagePath='stop.png'

# 识别屏幕上的图片位置并点击
def clickStart():
    start_gui=pyautogui.locateCenterOnScreen(startImagePath,confidence=0.8)# 找到匹配图像，返回图像中心坐标
    # 如果成功找到
    if start_gui!=None:
        print(start_gui)
        pyautogui.click(start_gui.x, start_gui.y, button='left')  # 单击左键
        return
    # 如果没有找到则找到暂停按钮单击暂停按钮中心
    stop_gui=pyautogui.locateCenterOnScreen(stopImagePath,confidence=0.8)# 找到匹配图像，返回图像中心坐标
    if stop_gui!=None:
        print(stop_gui)
        pyautogui.click(stop_gui.x, stop_gui.y, button='left')  # 单击左键

keyboard.add_hotkey('space', clickStart) #当按下热键时会执行对应函数
keyboard.wait('esc')# 按下esc键结束程序
