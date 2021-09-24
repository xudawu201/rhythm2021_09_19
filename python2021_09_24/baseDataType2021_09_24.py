'''
Author: xudawu
Date: 2021-09-24 09:29:27
'''
class baseDataType2021_09_24():
    def baseDataType(self):  # self，表示创建的类实例本身
        # 1
        t1 = (1, 2, 'R', 'py', 'Matlab');#元组
        list1 = list();#空列表

        # 2
        i = 0
        while i<len(t1):
            list1.append(t1[i])#元组元素写入空列表
            i = i+1

        # 3
        dict1 = dict()#空字典
        Li = {'a':'K', 'b':[3, 4, 5],'c':(1, 2, 6), 'd':18, 'e':50}#字典
        for LiElements in Li:
            dict1.setdefault(LiElements,Li[LiElements])#dict.setdefault(key, default=None)
                                               #如果键不存在于字典中，将会添加键并将值设为default

        # print
        print('t1:',t1)
        print('list1:', list1)
        print('Li:', Li)
        print('dict1:', dict1)

        #2.1
        import math

        def computeCylinder(radius, high):  # self，表示创建的类实例本身
            # def computeCylinder(self,radius, high):  # self，表示创建的类实例本身,
            # 母函数已定义了self,子函数不加，不然会被认为self也是参数
            #self.radius = radius #self指在类方法内部的参数
            superficialArea = 2 * math.pi ** radius + 2 * math.pi * radius * high
            volume=2*math.pi**radius*high
            return [superficialArea, volume]  #返回列表
            #return #若不带return则是传值，不会改变传参变量的值，return默认返回none

        cylinderResult = computeCylinder(10,11)#接收列表
        print('Cylinder superficial Area is:',cylinderResult[0])
        print('Cylinder volume is:', cylinderResult[1])
