'''
Author: xudawu
Date: 2021-10-23 16:16:45
LastEditors: xudawu
LastEditTime: 2021-10-23 16:19:42
'''
'''
Author: xudawu
Date: 2021-10-21 22:02:32
LastEditors: xudawu
LastEditTime: 2021-10-23 16:12:44
'''
#utf-8
import os
import sys

def datToTxt():
    path_0 = r"D:"
    path_1 = r"D:" + '\\'
    sys.path.append(path_1)
    #print(sys.path)
    #list current all files
    files = os.listdir(path_0)
    print('files', files)
    for filename in files:
        portion = os.path.splitext(filename)
        if portion[1] == ".dat":
            # recombine file name
            newname = portion[0] + ".txt"
            filenamedir = path_1 + filename
            newnamedir = path_1 + newname
            #os.rename(filename, newname)
            os.rename(filenamedir, newnamedir)
def datToCsv():
    path_0 = r"D:"
    path_1 = r"D:"
    filelist = os.listdir(path_0)
    for files in filelist:
        dir_path = os.path.join(path_0, files)
        #分离文件名和文件类型
        file_name = os.path.splitext(files)[0]  #文件名
        file_type = os.path.splitext(files)[1]  #文件类型
        print(dir_path)
        file_test = open(dir_path, 'r')
        #将.dat文件转为.csv文件
        new_dir = os.path.join(path_1, str(file_name) + '.csv')
        print(new_dir)
        file_test2 = open(new_dir, 'w')
        for lines in file_test.readlines():
            str_data = ",".join(lines.split('\t'))
            file_test2.write(str_data)
        file_test.close()
        file_test2.close()