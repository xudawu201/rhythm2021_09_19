'''
Author: xudawu
Date: 2022-03-06 18:54:34
LastEditors: xudawu
LastEditTime: 2022-03-08 22:20:20
'''
import pyautogui#鼠标键盘控制

#pyautogui.PAUSE = 1  #停顿功能，意味着所有pyautogui的指令都要暂停一秒；其他指令不会停顿；这样做，可以防止键盘鼠标操作太快；

#获取显示器分辨率
print(pyautogui.size())  # 返回所用显示器的分辨率； 输出：Size(width=1920, height=1080)
width, height = pyautogui.size()
print(width, height)  # 1920 1080
print('- '*10,'分割线','- '*10)
#1.鼠标操作
#移动鼠标
# pyautogui.moveTo(500, 300)
pyautogui.moveTo(500, 300, duration=1)#移动到(x,y)坐标，移动用时
#相对移动
pyautogui.moveRel(100, 300, duration=1)  # 第一个参数是左右移动像素值，第二个是上下，每个pyautogui都有duration参数，为可选，默认为0
#获取鼠标位置
print(pyautogui.position())  # 得到当前鼠标位置；输出：Point(x, y)
print('- ' * 10, '分割线', '- ' * 10)
#单击鼠标
# pyautogui.click(500, 300)  # 鼠标点击指定位置，默认左键
pyautogui.click(500, 300, button='left')  # 单击左键
# pyautogui.click(1000, 300, button='right')  # 单击右键
# pyautogui.click(1000, 300, button='middle')  # 单击中间
#双击鼠标
pyautogui.doubleClick(500, 300)  # 指定位置，双击左键
# pyautogui.rightClick(10, 10)  # 指定位置，双击右键
# pyautogui.middleClick(10, 10)  # 指定位置，双击中键
#鼠标点击与释放
pyautogui.mouseDown()  # 鼠标按下
pyautogui.mouseUp()  # 鼠标释放
#鼠标绝对拖动
# pyautogui.dragTo(100, 300, duration=1)
#鼠标相对拖动
# pyautogui.dragRel(100, 500, duration=1)  # 第一个参数是左右移动像素值，第二个是上下，
#滚动鼠标
pyautogui.scroll(-500)  # 滚动鼠标；

#2.屏幕处理
'''
获取屏幕截图
im = pyautogui.screenshot()：返回屏幕的截图，是一个Pillow的image对象
im.getpixel((500, 500))：返回im对象上，（500，500）这一点像素的颜色，是一个RGB元组
pyautogui.pixelMatchesColor(500,500,(12,120,400)) ：是一个对比函数，对比的是屏幕上（500，500）这一点像素的颜色，与所给的元素是否相同；
'''
# im = pyautogui.screenshot()
# im.save('screenshot.png')#相对路径
# im.save(r'C:\Users\xudaw\Desktop\screenshot.png')#指定存储绝对路径

#识别图像
'''
首先，我们需要先获得一个屏幕快照，例如我们想要点赞，我们就先把大拇指的图片保存下来；然后使用函数：locateOnScreen(‘zan.png’) ，
如果可以找到图片，则返回图片的位置，如：Box(left=25, top=703, width=22, height=22)；如果找不到图片，则返回None;
如果，屏幕上有多处图片可以匹配，则需要使用locateAllOnScreen(‘zan.png’) ，如果匹配到多个值，则返回一个list，参考如下：
'''
# 图像识别（一个）
btm = pyautogui.locateOnScreen('Test2021_10_13\image_recognition.PNG')
print('image_recognition:',btm)  # Box(left=, top=, width=, height=)
print('- ' * 10, '分割线', '- ' * 10)
# 图像识别（多个）
# btm = pyautogui.locateAllOnScreen('zan.png')
# print(list(btm))  # [Box(left=, top=, width=, height=), Box(left=, top=, width=, height=)]
# pyautogui.center((300, 500, 60, 60)) #返回指定位置的中心点；这样，我们就可以再配合鼠标操作点击找到图片的中心；
# pyautogui.locateCenterOnScreen('image_recognition.PNG')# 找到匹配图像，返回图像中心坐标

#3.键盘控制
'''
pyautogui.keyDown()#模拟按键按下；
pyautogui.keyUp()#模拟按键释放；
pyautogui.press()# 就是调用keyDown() & keyUp(),模拟一次按键；
pyautogui.typewrite('this',0.5)#第一参数是输入内容，第二个参数是每个字符间的间隔时间；
pyautogui.typewrite(['T','h','i','s'])#typewrite 也可以传入单字母的列表；
'''
#键盘输入函数
# pyautogui.keyDown('shift')  # 按下shift
# pyautogui.press('4')  # 按下 4
# pyautogui.keyUp('shift')  # 释放 shift
pyautogui.keyDown('down')
pyautogui.keyUp('down')
#按下组合键
pyautogui.hotkey('win', 'r')  # 按下win+r，按传入顺序按下，再按照相反顺序释放
#输出内容
pyautogui.typewrite(message='hello python', interval=0.05)#输出hello python 用时0.05s每个字符，输出不了中文
pyautogui.press('esc')
# pyautogui.alert(text='hello', title='alert')#提示框
#选择框
# a = pyautogui.confirm('选择一项', buttons=['A', 'B', 'C'])
# print(a)

#4.实例
#鼠标画正方形
for i in range(2):  # 画正方形
    pyautogui.moveTo(500, 300, duration=1)
    pyautogui.moveTo(500, 600, duration=1)
    pyautogui.moveTo(900, 600, duration=1)
    pyautogui.moveTo(900, 300, duration=1)

x, y = pyautogui.position()
rgb = pyautogui.screenshot().getpixel((x, y))
print('x:',x,' y:',y,' RGB:',rgb)