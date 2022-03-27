'''
Author: xudawu
Date: 2022-03-27 17:52:19
LastEditors: xudawu
LastEditTime: 2022-03-27 17:52:20
'''
a='abcdb'
#replace() 方法把字符串中的 old（旧字符串） 替换成 new(新字符串)，如果指定第三个参数max，则替换不超过 max 次。
# str.replace(old, new,max)
b=a.replace('b','',1)
print(b)