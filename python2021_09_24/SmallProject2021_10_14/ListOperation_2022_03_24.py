'''
Author: xudawu
Date: 2021-10-30 16:19:50
LastEditors: xudawu
LastEditTime: 2022-03-24 12:55:49
'''
a = [1, 6, 9]
print('a:', a, 'a[0]:', a[0])

# 1.append()
# append()对于列表的操作主要实现的是在特定的列表最后添加一个元素，并且只能一次添加一个元素，并且只能在列表最后；
# m.append(元素A)
a.append(2)
print('append', a)

# 2.extend()
# extend()对于列表的操作主要实现的是对于特定列表的扩展和增长，可以一次添加多个元素，不过也只能添加在列表的最后；
# m.extend([元素A，元素B，……])
a.extend([1, 2, 3])
print('extend', a)

# 3.m.insert(i,元素B)：表示在列表m里面的i处加入元素B
a.insert(1, 8)
print('insert', a)

# 4.m.remove的作用是移除掉列表m里面的特定元素；
# m.remove(元素A)
a.remove(8)
print('remove', a)

# 5.pop()默认弹出最后一位
a.pop()
print('pop', a)
# 指定索引弹出
a.pop(3)
print('pop(3)', a)

# count(a)输出元素a在列表m里面出现的次数
b = a.count(1)
print('count', b)

# m.index(A)：输出元素A在列表m里面第一次出现的索引位置号
b = a.index(2)
print('index', b)

# 列表拷贝,
# 深拷贝,不改变被拷贝的列表
b = a.copy()  #也可以是b=a[:]
b.append(3)
print(b)
print(a)

# 浅拷贝,会改变被拷贝的列表,本质是给a对象命名为b
b = a
b.append(3)
print(b)
print(a)