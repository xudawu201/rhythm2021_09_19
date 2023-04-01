'''
Author: xudawu
Date: 2023-03-31 17:12:36
LastEditors: xudawu
LastEditTime: 2023-04-01 17:33:18
'''
# import win32gui
# win32gui整合到pywin32中了
# win32gui Windows图形界面接口模块。主要负责操作窗口切换以及窗口中元素id标签的获取
from win32 import win32gui
# win32api提供了常用的用户API,Windows开发接口模块。
# 主要负责模拟键盘和鼠标操作,对win32gui获取的标签进行点击/获取值/修改值等操作
from win32 import win32api
# win32con全面的库函数，提供Win32gui和Win32api需要的操作参数
import win32con

# 获取最前窗口句柄
hwnd_int = win32gui.GetForegroundWindow()
print(hwnd_int)
# 获取坐标处的窗口句柄
hwnd_int = win32gui.WindowFromPoint((100, 100))
print(hwnd_int)
# 根据窗口标题，取得窗口句柄
hwnd_int = win32gui.FindWindow(0, '')
print(hwnd_int)
# 根据窗口类名，取得窗口句柄
# hwnd_int = win32gui.FindWindow(ClassName, Title)
# print(hwnd_int)

# 参数1 窗口类名    # 参数2 窗口标题--必须完整；如果该参数为None，则为所有窗口全匹配
# 返回值：如果函数成功，返回值为窗口句柄；如果函数失败，返回值为0

# 检测当前句柄是否存在  存在则返回  1  不存在返回 0
print('分隔符',' #'*10)
N = win32gui.IsWindowEnabled(133384)
print(N)
S = win32gui.IsWindowVisible(133384)
print(S)
V = win32gui.IsWindow(133384)
print(V)

# 获取窗口信息
print('分隔符',' #'*10)
# 获取窗口句柄 
hwnd_int = win32gui.WindowFromPoint((100, 100))
print(hwnd_int)
# 获取窗口标题 
windowsTitle_str = win32gui.GetWindowText(hwnd_int)
print(windowsTitle_str)
# 获取窗口类名 
windowsClassName_str = win32gui.GetClassName(hwnd_int)
print(windowsClassName_str)
# 返还窗口信息（x,y坐标，还有宽度，高度)是一个元组
windowsInfo_tuple = win32gui.GetWindowRect(hwnd_int)
print(windowsInfo_tuple)
print(windowsInfo_tuple[0])

print('分隔符',' #'*10)
#检查窗口是否最小化     #是返回1，不是返回0
i=win32gui.IsIconic(hwnd_int)
print(i)
# 激活该窗口,此时窗口会是最前面一层
# win32gui.SetForegroundWindow(hwnd_int)

# 找到窗口,0表示桌面
# win32gui.FindWindowEx(parentHwnd,Child, ClassName, Title)
# hwnd_int=win32gui.FindWindowEx(0,0, 0, '窗口标题')
# print(hwnd_int)

# 移动窗口
# win32gui.MoveWindow(hwnd_int,int X,int Y,int nWidth, int nHeight, BOOL bRepaint )
# win32gui.MoveWindow(hwnd_int,300,500,600,700,True)
# 移动某窗口hld到指定位置。x,y指与屏幕左上角距离，nWidth, nHeight 指长和高bRepaint ：是否重绘

# 鼠标操作
print('分隔符',' #'*10)
# 获取当前鼠标的坐标
mousePos_tuple =win32gui.GetCursorPos()
print(mousePos_tuple)
# 也用win32api获取鼠标位置
mousePos_tuple=win32api.GetCursorPos()
print(mousePos_tuple)
# 移动鼠标位置
win32api.SetCursorPos((500,500))
#按下鼠标左键
# win32api.mouse_event(键位操作, 0, 0)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
#松开鼠标左键
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
'''
键位表
MOUSEEVENTF_LEFTDOWN:表明接按下鼠标左键
MOUSEEVENTF_LEFTUP:表明松开鼠标左键
MOUSEEVENTF_RIGHTDOWN:表明按下鼠标右键
MOUSEEVENTF_RIGHTUP:表明松开鼠标右键
MOUSEEVENTF_MIDDLEDOWN:表明按下鼠标中键
MOUSEEVENTF_MIDDLEUP:表明松开鼠标中键
MOUSEEVENTF_WHEEL:鼠标轮移动,数量由data给出
'''
print('分隔符',' #'*10)
# 键盘操作
# win32api.keybd_event(键码, 硬件扫描码, 按下还是弹起的标志位, 与击键相关的附加的32位值)
win32api.keybd_event(65, 0, 0, 0)
# 松开A
win32api.keybd_event(65, 0, 1, 0)
# 操作键盘使用

# 进行截图
'''
Author: xudawu
Date: 2023-03-31 21:24:19
LastEditors: xudawu
LastEditTime: 2023-04-01 17:32:15
'''
import win32gui
import win32api
import win32ui
import win32con
import numpy as np
from PIL import Image
from ctypes import windll
# 操作图像
from PIL import ImageGrab
import pyautogui
import time

# 使用pywin32截图时间0.003秒
# 使用ImageGrab.grab截图时间0.04秒
# 使用pyautogui截图时间0.45秒
class GetWindowsScreenshot():
    def __init__(self,):
        pass
    def getWindowsScreenshot(self,hwnd,region=None):
        a=time.time()
        # 获取窗口位置、大小
        if region!=None:
            # 截取窗口区域,其实xy坐标和截取的长度和高度
            left, top, width, height = region
        else:
            # 如果没有指定截取大小,则获取屏幕区域
            # 用于获取屏幕宽度
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            # 获取屏幕高度
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            # 获取屏幕左边距
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            # 获取屏幕右边距
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
        # 激活窗口到最上层
        # win32gui.SetForegroundWindow(hwnd)
        # 截图区域
        # img = ImageGrab.grab(region)
        # img = pyautogui.screenshot(region=[0,0,1000,500])
        # 获取窗口设备上下文DC
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)

        # 创建内存设备上下文DC
        save_dc = mfc_dc.CreateCompatibleDC()

        # 创建位图对象
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)

        # 将位图选入内存设备上下文DC,即将位图数据放置在刚开辟的内存里
        save_dc.SelectObject(save_bitmap)

        # 如果窗口使用了硬件加速渲染，那么使用位图方式截取窗口截图可能会出现黑屏问题。
        # 在这种情况下，可以尝试使用`PrintWindow`函数来捕获屏幕上的窗口图像，即使窗口使用了硬件加速也可以捕获。
        
        # 从设备上下文DC中复制位图数据到内存设备上下文DC中
        # save_dc.BitBlt((从窗口的xy坐标开始截图), (要截取的长和高), mfc_dc, (相对xy坐标的起始点), win32con.SRCCOPY)
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (left, top), win32con.SRCCOPY)
        
        # 调用PrintWindow函数捕获窗口图像并保存到位图对象中
        # 如果窗口使用了硬件加速渲染，则PrintWindow函数可以正确捕获图像
        # 0-保存整个窗口，1-只保存客户区,成功保存返回1
        # result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
        # print('result: ',result)

        # # 保存图片到本地,是未处理的彩色
        # save_bitmap.SaveBitmapFile(save_dc, 'screenshot.png')

        # 获取位图数据
        bmpinfo = save_bitmap.GetInfo()
        bmp_str = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmp_str, 'raw', 'BGRX', 0, 1)
    
        # img = np.frombuffer(bmp_str, dtype='uint8')
        # img.shape = (height, width, 4)
        # img = img[:, :, :3]
        # img = Image.fromarray(img)

        # 释放设备上下文DC
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.DeleteObject(save_bitmap.GetHandle())
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # # # 转换图像格式，并返回灰度图片
        # # img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        b=time.time()
        print(b-a)
        return img

# 获取窗口截图并显示
# Geometry Dash窗口
# hwnd_int = win32gui.FindWindow(0, 'Geometry Dash')
hwnd_int = win32gui.WindowFromPoint((100, 100))
print(hwnd_int)
# 获取窗口标题 
windowsTitle_str = win32gui.GetWindowText(hwnd_int)
print(windowsTitle_str)
# 获取窗口位置
print(win32gui.GetWindowRect(hwnd_int))
# 指定截图区域,起始xy坐标和截取的长和高
region=(0,0,1000,500)
GetWindowsScreenshot=GetWindowsScreenshot()
# img = GetWindowsScreenshot.getWindowsScreenshot(hwnd_int)
img = GetWindowsScreenshot.getWindowsScreenshot(hwnd_int,region)
# img.show()