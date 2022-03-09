'''
Author: xudawu
Date: 2021-10-14 10:24:59
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x**2  #x的平方

plt.figure()
'''
plt.plot(x, y, ls='-', lw=2, label='xxx', color='g' )
x： x轴上的值
y： y轴上的值
ls：线条风格 (linestyle)
lw：线条宽度 (linewidth)
label：标签文本
'''

plt.plot(x, y1,)  #画线,默认颜色为blue,
#plt.text(x, y, s, fontsize, verticalalignment,horizontalalignment,rotation , **kwargs)
point_point ={'x1':[5,3],'x2':[4,9],'x3':[2,2],'x4':[1,7]}
plt.text(point_point.get('x1')[0], point_point.get('x1')[1], 'test_label', fontsize=10, verticalalignment='top', horizontalalignment='right', rotation=0)
# plt.scatter(5, 3, color='red')  #画点s
for item in point_point.values():
    plt.scatter(item[0],item[1])
# plt.scatter(5, 3, color='red')  #画点s

#两点之间连线
p1 = [point_point.get('x2')[0], point_point.get('x2')[1]]  #点p1的坐标值
p2 = [point_point.get('x3')[0], point_point.get('x3')[1]]  #点p2的坐标值
# plt.plot([x1, x2], [y1, y2])  #简单理解就是：先写x的取值范围，再写y的取值范围
plt.plot([point_point.get('x2')[0],point_point.get('x3')[0]],
         [point_point.get('x2')[1],point_point.get('x3')[1]],'--')

plt.plot(x, y2,'--')#第二条线默认颜色会自动更换,'-'实线,'--'虚线
# plt.savefig('test.jpg')#存储图片
plt.show()