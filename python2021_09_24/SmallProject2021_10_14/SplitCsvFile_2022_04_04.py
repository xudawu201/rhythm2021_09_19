'''
Author: xudawu
Date: 2022-04-04 18:49:16
LastEditors: xudawu
LastEditTime: 2022-04-04 18:57:25
'''
import pandas
filePath=r'D:\UpUp_2019_06_25\vscode_2020_01_07\python_2021_09_17\SomeResource2021_10_23\split_test.csv'
keyName='label'
#分割csv文件
def splitCsv(filePath,keyName):
    #读取csv文件
    fileRead_DataFrame=pandas.read_csv(filePath,encoding='utf-8')
    print(fileRead_DataFrame)
    #将csv分组
    fileReadGroup_DataFrame=fileRead_DataFrame.groupby(keyName)
    #每组分别存储
    for i in range(len(fileReadGroup_DataFrame)):
        #获取每组的组名
        curGroupName_str=fileReadGroup_DataFrame.size().index[i]
        #得到当前组
        curGroup_DataFrame=fileReadGroup_DataFrame.get_group(curGroupName_str)
        tempFileName=keyName+str(i)+'.csv'
        #存储文件
        curGroup_DataFrame.to_csv(tempFileName,encoding='utf-8',header=False, index=False)

splitCsv(filePath,keyName)