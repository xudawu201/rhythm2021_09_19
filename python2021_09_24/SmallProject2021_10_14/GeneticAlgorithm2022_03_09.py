'''
Author: xudawu
Date: 2022-03-09 14:56:00
LastEditors: xudawu
LastEditTime: 2022-03-09 15:16:09
'''
import random
'''
问题描述：给定目标句子，例如目标句为Genetic algorithm is a search algorithm used in computational mathematics to solve optimization.，
由随机字符组成的句子演化成目标句。
'''
#geneSet表示所有包含的字符组成的集合
geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
# target表示目标字符串。
target = "Genetic algorithm is a search algorithm used in computational mathematics to solve optimization."


#生成初始个体
# 接下来需要生成初始个体，即由字符集合随机生成一个与目标字符串长度相等的字符串。
def generateParent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
        #list1=extend(list2) 函数用于在列表末尾一次性追加另一个序列中的多个值（将list2的值追加到list1）。
        #random.sample(list,range) 随机选取list中的range个值
    return ''.join(
        genes
    )  #'sep'.join(str)将序列中的元素以指定的字符连接生成一个新的字符串,sep为分隔符,可以为空,str为要连接的字符序列


# 适应度
# 遗传算法提供的适应度值是引擎获得的唯一反馈，可以引导其走向一个解决方案。在这个问题中，适应度值为当前字符串与目标字符串匹配的字符个数。
def getFitness(guess, target):
    # zip()返回：一个zip对象，其内部元素为元组；从参数中的多个迭代器取元素组合成一个新的迭代器
    sum = 0
    for actual, expected in zip(guess, target):
        if actual == expected:
            sum = sum + 1
    return sum


# 变异
# 将字符串的任意1个位置字符调换，即可完成变异操作
def mutate(parent, geneSet):
    #随机选择一个位置
    idx = random.randrange(0, len(parent))
    childGenes = list(parent)
    #随机选取两个字符
    newGene= random.sample(geneSet,1)
    childGenes[idx] = newGene[0]
    return ''.join(childGenes)


# 获取已使用时间
def getTimeUsed(startTime_time):
    import time
    from datetime import timedelta
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    return timedelta(seconds=int(round(time_dif)))


# 可视化过程
def showEvolution(epoch,guess, startTime):
    fitness = getFitness(guess, target)
    timeUsed = getTimeUsed(startTime)
    print('epoch:',epoch,':',guess,' fitness:',fitness,'timeUsed:',timeUsed )


# random.seed()


#进化函数
def getEvolutionGene(geneSet, target):
    #记录时间戳
    import time
    startTime = time.time()
    #获取一个父个体
    parentGene = generateParent(len(target))
    #适应性算法
    parentFitness = getFitness(parentGene, target)
    #进化次数
    epoch=0
    while True:
        #变异
        childGene = mutate(parentGene, geneSet)
        #物竞
        childFitness = getFitness(childGene, target)
        #天择
        #如果子代不如父代,不执行循环体剩余部分,直接进行下次循环,重新选择变异基因
        epoch = epoch + 1#进化次数
        if parentFitness >= childFitness:
            continue
        #进化过程可视化
        showEvolution(epoch,childGene, startTime)
        #如果已达到目标,则退出进化
        if childFitness >= len(target):
            break
        #当前代繁衍下一代,迭代进化
        parentFitness = childFitness
        parentGene = childGene
    return parentGene
#开始进化演变
bestGene = getEvolutionGene(geneSet, target)
print('最好的基因是:',bestGene)
