'''
Author: xudawu
Date: 2022-03-18 13:25:35
LastEditors: xudawu
LastEditTime: 2022-03-18 13:28:03
'''
import time
from datetime import timedelta

# 获取已使用时间
def getTimeUsed(startTime_time):
    #获得时间差
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    #返回时分秒格式
    return timedelta(seconds=int(time_dif))

startTime_time=time.time()
time.sleep(2.3456)
usedTime=getTimeUsed(startTime_time)
print(usedTime)