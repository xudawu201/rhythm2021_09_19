'''
Author: xudawu
Date: 2021-10-13 17:04:25
'''

# for循环
print('for循环')
#类中方法里的属性要有self. 否则算是未定义变量
list_a = list()
for a in range(5):
    list_a.append(a)
    print(list_a)
        
# 列表推导式
print('# '*20)#分隔符号
print('列表推导式')
list_b = [b for b in range(5)]
print(list_b)

print('# ' * 20)  #分隔符号
print('in后面跟其他可迭代对象,如字符串')
# in后面跟其他可迭代对象,如字符串
print('列表推导式')
list_c = [7 * c for c in "python"]
print(list_c)

print('普通for循环式')
listTempC=list()
for tempC in 'python':
    listTempC.append(tempC*7)
print(listTempC)

print('# ' * 20)  #分隔符号
print('带if条件语句的列表推导式')
# 带if条件语句的列表推导式
list_d = [d for d in range(6) if d % 2 != 0]
print(list_d)