'''
Author: xudawu
Date: 2022-04-28 20:16:04
LastEditors: xudawu
LastEditTime: 2022-05-02 12:12:36
'''

import csv
import pandas as pd
a=[
    [1,2,3],
    [4,5],
    [23]
    ]

#存储CSV文件
def saveCsvFile(filePath,content_list):
    # 使用open()方法打开这个csv文件,newline=''表示不换行存储,w重写
    fileOpen = open(filePath, mode='w', encoding='utf-8-sig',newline='')
    # 写入csv文件
    fileOpenCsv=csv.writer(fileOpen)
    # writerows()将一个二维列表中的每一个列表写为一行。
    fileOpenCsv.writerows(content_list)
    # 关闭文件
    fileOpen.close()
    print('save file success')

# pandas存储文件,存储为一列
def pandasSaveCsvFile(filePath,content_list):
    df=pd.DataFrame(data=content_list)
    # sep=' ' 将空格作为分隔符,每一个列表在一个格子里，而不是每个列表的数分别在一个格子里
    df.to_csv(filePath,sep=' ',header=False,index=False)
    print('save file success')

#读取csv文件返回list
def openCsvFile(filePath):
    # 使用open()方法打开这个csv文件
    fileOpen=open(filePath, mode='r', encoding='utf-8-sig')
    # 在用csv.reader()方法读取这个文件
    file_csv = csv.reader(fileOpen)
    # 转为list
    file_list=list(file_csv)
    print('read file success')
    return file_list
    
#去除含有指定内容的行
def removeSpecificLines(filePath,column_int,column_str):
    #存储CSV文件
    def saveCsvFile(filePath,content_list):
        # 使用open()方法打开这个csv文件,newline=''表示不换行存储,w重写
        fileOpen = open(filePath, mode='w', encoding='utf-8-sig',newline='')
        # 写入csv文件
        fileOpenCsv=csv.writer(fileOpen)
        # writerows()将一个二维列表中的每一个列表写为一行。
        fileOpenCsv.writerows(content_list)
        # 关闭文件
        fileOpen.close()
        print('save file success')
    
    # 使用open()方法打开这个csv文件
    fileOpenRead=open(filePath, mode='r', encoding='utf-8-sig')
    # 在用csv.reader()方法读取这个文件
    fileReader_csv = csv.reader(fileOpenRead)
    #临时列表存储文件内容
    tempFile_list=[]
    for row in fileReader_csv:
        print(row)
        if row[column_int] != column_str:
            tempFile_list.append(row)
    # 关闭文件
    fileOpenRead.close()
    # 新文件名字
    saveFilePath=filePath.replace('.csv','Remove.csv')
    # 调用存储函数
    saveCsvFile(saveFilePath,tempFile_list)
    print('remove file success')

filePath='testCsv.csv'
saveCsvFile(filePath,a)

fileContent_list=openCsvFile(filePath)
print(fileContent_list)

removeSpecificLines(filePath,0,'')

# pandasSaveCsvFile(filePath,a)
