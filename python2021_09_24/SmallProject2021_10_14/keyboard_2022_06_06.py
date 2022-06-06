'''
Author: xudawu
Date: 2022-06-06 16:57:25
LastEditors: xudawu
LastEditTime: 2022-06-06 16:57:32
'''
import keyboard #监听键盘

keyboard.wait('a') #一直等到按下a键才会执行下面的程序
print('a')
# keyboard.wait()#没有设置按键会一直监听下去
def test_a():
    print('test_a')
def test_b(x_int):
    print('test_b:',x_int)

keyboard.add_hotkey('a', test_a) #当按下热键时会执行对应函数
keyboard.add_hotkey('b', test_b, args=(1,)) #当按下热键时会执行对应函数，并且传递参数
# keyboard.wait('esc') # 持续监听上面的热键

# recorded = keyboard.record(until='esc')#记录键盘事件，如果加上until参数，可以设置当按下某按键时结束监听，和wait方法有点像
# print(recorded)
# keyboard.play(recorded, speed_factor=3)#播放记录的键盘事件