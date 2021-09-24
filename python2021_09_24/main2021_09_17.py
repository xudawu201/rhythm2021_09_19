'''
Author: xudawu
Date: 2021-09-17 10:24:12
'''
from baseDataType2021_09_24 import baseDataType2021_09_24  # form 文件名 import 类名

while True:
    print('输入数字选择功能序号')
    print('0.退出程序')
    print('1.类&函数测试')
    selection = int(input())#将输入转为int型，if条件才能判断
    if selection == 0:
        break
    if selection == 1:
        baseDataTypeTest = baseDataType2021_09_24()  # 实例化类需要带括号
        #python的类，带括号是实例化，不带括号是赋值。
        baseDataTypeTest.baseDataType()
