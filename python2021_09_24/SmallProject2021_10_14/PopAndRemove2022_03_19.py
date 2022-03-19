'''
Author: xudawu
Date: 2022-03-19 13:39:23
LastEditors: xudawu
LastEditTime: 2022-03-19 13:41:40
'''
a=[5,4,3,2,1]
print(a)
#pop()函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
b=a.pop(1)
print(b)
print(a)
#remove() 函数用于移除列表中某个值的第一个匹配项。
a.remove(1)
print(a)