'''
Author: xudawu
Date: 2022-03-29 15:57:54
LastEditors: xudawu
LastEditTime: 2022-03-31 12:40:32
'''
from datetime import timedelta
import multiprocessing
import time

def getUsedTime(startTime_time):
    endTime_time=time.time()
    duringTime_time=endTime_time-startTime_time
    timeUsed_time=timedelta(seconds=int(round(duringTime_time)))
    return timeUsed_time

def testDef(num_int):
    time.sleep(2)
    print('process:',num_int)
    return 'test'

def main():
    # Process类的构造方法：
    # init(self, group=None, target=None, name=None, args=(), kwargs={})
    # 参数说明：
    # group：进程所属组，基本不用。
    # target：表示调用对象，一般为函数。
    # args：表示调用对象的位置参数元组,只有一个参数也要加逗号
    # name：进程别名。
    # kwargs：表示调用对象的字典。
    process1=multiprocessing.Process(target=testDef,args=(1,))
    process2=multiprocessing.Process(target=testDef,kwargs={'num_int':2})
    #进程开始
    startTime_time=time.time()
    process1.start()
    process2.start()
    #等待进程完成
    process1.join()
    process2.join()

    timeUsed_time=getUsedTime(startTime_time)
    print('timeUsed_time:',timeUsed_time)

    #1.串行进程
    startTime_time=time.time()
    for i in range(3,8):
        testDef(i)
    timeUsed_time=getUsedTime(startTime_time)
    print('timeUsed_time:',timeUsed_time)

    #2.并行进程
    #创建进程池
    multiPool_pool=multiprocessing.Pool(processes=5)

    #单个进程,一次只能一个
    multiPool_pool.apply_async(testDef,args=(3,))

    #进程池关闭
    # multiPool_pool.close()
    # multiPool_pool.join()

    #多个进程
    # map_async()是异步的,主程序不会等待进程完成而直接往下运行,multiPool_pool.wait()可以实现等待进程完成再执行下面的程序
    # map()是阻塞的,map(func, iterable[, chunksize=None])
    # 虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程；
    # map返回的是一个列表，由func函数的返回值组成
    # multiPool_pool.map_async()
    startTime_time=time.time()
    a=multiPool_pool.map(testDef,[4,5,6,7])
    print(a)
    multiPool_pool.close()
    multiPool_pool.join()
    timeUsed_time=getUsedTime(startTime_time)
    print('timeUsed_time:',timeUsed_time)
if __name__ == '__main__':
    main()