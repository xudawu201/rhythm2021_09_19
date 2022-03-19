'''
Author: xudawu
Date: 2022-03-19 12:58:25
LastEditors: xudawu
LastEditTime: 2022-03-19 13:01:59
'''
#函数写法
def lambdaTest(x,y):
    return x+y

a=lambdaTest(1,2)
print(a)

#lambda写法 a=lambda x,y:x+y  a为函数名,x,y为函数输入值,:号后的x+y为返回值
b=lambda x,y:x+y
c = b(1, 2)
print(c)