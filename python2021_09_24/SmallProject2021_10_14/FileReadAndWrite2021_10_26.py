'''
Author: xudawu
Date: 2021-10-26 15:55:11
LastEditors: xudawu
LastEditTime: 2021-10-27 13:32:11
'''
class FileReadAndWrite2021_10_26():
    #存入txt文件，以追加方式
    def saveInTxtFileByAppend_txt(self,fileName,content):
        fileNovel_txt = open(fileName, mode='a', encoding='utf-8')
        fileNovel_txt.write(content)  #write 写入
        fileNovel_txt.close()  #关闭文件

    def getInTxtFileByRead_txt(self, fileName):
        fileNovel_txt = open(fileName, mode='r', encoding='utf-8')
        fileContent_txt=fileNovel_txt.read()  #读取
        fileNovel_txt.close()  #关闭文件
        return fileContent_txt