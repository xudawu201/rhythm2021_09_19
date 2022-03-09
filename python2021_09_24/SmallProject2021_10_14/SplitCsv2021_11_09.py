'''
Author: xudawu
Date: 2021-11-09 09:40:16
LastEditors: xudawu
LastEditTime: 2021-11-09 11:48:19
'''
import time
from datetime import timedelta
# 获取已使用时间
def getTimeUsed(startTime_time):
    endTime_time = time.time()
    time_dif = endTime_time - startTime_time
    return timedelta(seconds=int(round(time_dif)))

#分类切分csv文件
class SplitCsv():
    def __init__(self):
        self.none=0

    #保存csv文件
    def saveWordCutForCsv(self, filePath, contentCut_list):
        import csv
        starTime_time = time.time()
        fileOpen = open(filePath, mode='w', encoding='utf-8', newline='')
        fileOpenCsv = csv.writer(fileOpen)

        fileOpenCsv.writerows(contentCut_list)

        print('csv file save complete,time used:', getTimeUsed(starTime_time))
        fileOpen.close()


    def splitCsv(self,filePath,labelName):
        starTime_time = time.time()
        import pandas as pd
        import csv
        filePath_path = filePath
        fileOpen_Dataframe=pd.read_csv(filePath_path,encoding='utf-8')

        content_list=[]
        Allcontent_list=[]
        tempLabelName = fileOpen_Dataframe[labelName][0]
        contentGroup=fileOpen_Dataframe.groupby([labelName])
        for item in contentGroup.get_group('体育').iterrows():
            content_list.append(list(item[0]))
        print(content_list)

        # print('read file', labelName, 'contents complete,time used:', getTimeUsed(starTime_time))


        return contentGroup.get_group('体育')


filePath_path = 'SmallProject2021_10_14/test.csv'
labelName='label'

splitCsv = SplitCsv()
content_list=splitCsv.splitCsv(filePath_path, labelName)
splitCsv.saveWordCutForCsv('SmallProject2021_10_14/test01.csv', content_list)
