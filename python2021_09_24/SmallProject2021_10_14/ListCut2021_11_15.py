'''
Author: xudawu
Date: 2021-11-15 15:14:43
LastEditors: xudawu
LastEditTime: 2021-11-15 15:23:12
'''
a = [1, 2, 3, 4, 5]
b = a[1:3]  #[i:j]切分列表，返回从i到j不包含j的列表
print(b)
b = a[-1]  #返回列表最后一个元素
print('- ' * 5, 'split line', '- ' * 5)
print(b)
b = a[0:]  #默认为全切分
print('- ' * 5, 'split line', '- ' * 5)
print(b)
b = a[:4:2]  #[i:j:k]切分列表，返回从i到j不包含j，步长为k的列表
print('- ' * 5, 'split line', '- ' * 5)
print(b)
c=[[1,2,3],[8,9,3],[3,7,4]]
b=c[0:1]#[i:j]切分，返回从i到j不包含j的列表
print('- ' * 5, 'split line', '- ' * 5)
print(b)
'''
[2, 3]
- - - - -  split line - - - - - 
5
- - - - -  split line - - - - - 
[1, 2, 3, 4, 5]
- - - - -  split line - - - - - 
[1, 3]
- - - - -  split line - - - - - 
[[1, 2, 3]]
'''