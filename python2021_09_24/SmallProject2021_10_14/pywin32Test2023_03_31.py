'''
Author: xudawu
Date: 2023-03-31 17:12:36
LastEditors: xudawu
LastEditTime: 2023-03-31 19:09:14
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