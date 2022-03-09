'''
Author: xudawu
Date: 2021-10-26 17:44:34
LastEditors: xudawu
LastEditTime: 2021-10-26 17:49:01
'''
#!/usr/bin/python
#下面的程序会每隔5秒显示当前的日期和时间
import time
for i in range(3):
    ### 显示当前的日期和时间 ##
    print ("当前的日期 &amp; 时间 " + time.strftime("%c"))
    #### 延迟5秒执行 ####
    time.sleep(2)

import random
print (random.uniform(10, 20))#随机数
print(random.randint(10,20))#随机整数
