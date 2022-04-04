'''
Author: xudawu
Date: 2021-10-14 15:13:15
LastEditors: xudawu
LastEditTime: 2022-04-04 18:18:07
'''
import pandas

students = [
    ('mayiming', 23, 'A'), 
    ('xudawu', 23, 'B'), 
    ('xudawu', 24, 'A'), 
    ('like', 24, 'B')
    ] 
# 添加行列别名
# students_DataFrame = pandas.DataFrame(students, columns =['Name', 'Age', 'Section'], index =['1', '2', '3', '4']) 
students_DataFrame = pandas.DataFrame(students)
print(students_DataFrame)
print('- '*10,'分割线1','- '*10)

#遍历列标签
for item in students_DataFrame:
    print(item)
#分割线
print('- '*10,'分割线2','- '*10)

#遍历某列
for item in students_DataFrame:
    print(students_DataFrame[item])
print('- '*10,'分割线3','- '*10)

# loc是location的意思，和iloc中i的意思是指integer，所以它只接受整数作为参数，详情见下面。
# loc works on labels in the index.
# iloc works on the positions in the index (so it only takes integers).
# 获得行数据
a=[]
for item in students_DataFrame.loc[0]:
    a.append(item)
print(a)
print('- '*10,'分割线4','- '*10)

#取标签为0的行
print(students_DataFrame.loc[0])
print('- '*10,'分割线5','- '*10)

#取标签为0的行
print(students_DataFrame.loc[0,:])
print('- '*10,'分割线6','- '*10)

#取标签为0到2的行的标签为0的列
print(students_DataFrame.loc[0:2,0])
print('- '*10,'分割线7','- '*10)

#取标签为0的列
print(students_DataFrame.loc[:,0])
print('- '*10,'分割线8','- '*10)

studentsGroup_DataFrame = students_DataFrame.groupby(0)
curGroup=studentsGroup_DataFrame.get_group('xudawu')
print(curGroup)
print('- '*10,'分割线9','- '*10)

#数据框转为列表
import numpy
a=numpy.array(curGroup)
#先转为数组
print(a)
a=a.tolist()
print(a)