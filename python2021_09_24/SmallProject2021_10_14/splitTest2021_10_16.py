'''
Author: xudawu
Date: 2021-10-16 16:36:06
LastEditors: xudawu
LastEditTime: 2022-01-20 13:13:40
'''
'''
1、split()函数
语法：str.split(str="",num=string.count(str))[n]
参数说明：
str:表示为分隔符，默认为空格，但是不能为空('')。若字符串中没有分隔符，则把整个字符串作为列表的一个元素
num:表示分割次数。如果存在参数num，则仅分隔成 num+1 个子字符串，并且每一个子字符串可以赋给新的变量
[n]:表示选取第n个分片
注意：当使用空格作为分隔符时，对于中间为空的项会自动忽略
'''
# 1.以'.'为分隔符
string = "www.baidu .com.cn"
print(string.split('.'))

# 2.分割两次,分割两次之后之后的字符串不会再切分
print(string.split('.',2))

# 3.分割两次，并取序列为1的项
print(string.split('.',2)[1])

print('- '*10,'分割线','- '*10)

# 4.分割两次，并把分割后的三个部分保存到三个文件
u1, u2, u3 =string.split('.',2)
print(u1)
print(u2)
print(u3)