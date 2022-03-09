'''
Author: xudawu
Date: 2021-10-14 13:49:31
LastEditors: xudawu
LastEditTime: 2022-02-26 16:21:43
'''

import requests
from bs4 import BeautifulSoup
import time
import random


def getHtml_soup(url):
    #头部伪装
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
    }

    #获取html内容
    req = requests.get(url, headers=headers)
    req.encoding = 'gbk'
    html = req.text
    soup_html = BeautifulSoup(html, "html.parser")
    return soup_html

def getHtmlContent_str(html_soup, titleClassName, contentClassName):
    #存入字符串章节标题
    htmlTitle_str = html_soup.find(class_=titleClassName).h1.string
    #存入字符串小说内容
    htmlContent_str = htmlTitle_str + '\n' + html_soup.find(
        class_=contentClassName).get_text() +'\n' .replace(
            '(请记住本书首发域名：www.yqxs.cc。笔趣阁手机版阅读网址：m.yqxs.cc', '')
    #显示章节内容
    print(htmlContent_str)
    return htmlTitle_str, htmlContent_str

#存入txt文件，以追加方式
def saveInTxtFileByAppend_txt(fileName, content_str):
    fileNovel_txt = open(fileName, mode='a', encoding='utf-8')
    fileNovel_txt.write(content_str)  #write 写入
    fileNovel_txt.close()  #关闭文件

def getNextUrl_str(html_soup,urlPre,nextPageClassName):
    #选取特定标签
    # .next_sibling下一个兄弟节点,下一个节点可为空白和换行，继续查找下一个即可
    htmlContentSelect_soup = html_soup.find(class_=nextPageClassName).ul.li.next_sibling.next_sibling.next_sibling.next_sibling.a
    nextHtmlUrlAfter_str = htmlContentSelect_soup.get('href')
    nextHtmlUrl_str = urlPre + str(nextHtmlUrlAfter_str)
    return nextHtmlUrl_str


if __name__ == '__main__':
    fileName = 'htmlContent.txt'
    urlPre = 'https://www.yqxs.cc/'
    url = 'https://www.yqxs.cc/html/59000/59000330/10079983.html'
    titleClassName = 'content'
    contentClassName = 'showtxt'
    nextPageClassName = 'page_chapter'
    nextHtmlUrl_str = url
    for item in range(405):
        #获取soup
        htmlContent_soup = getHtml_soup(nextHtmlUrl_str)
        #获取文章标题
        # print(htmlContent_soup.title.string)
        # print(htmlContent_soup.find(class_='content').h1.string)
        # #获取文章内容
        # print(htmlContent_soup.find(class_='showtxt').get_text())
        #获取html内容
        htmlContentTitle_str, htmlContent_str = getHtmlContent_str(htmlContent_soup, titleClassName,contentClassName)
        #存储html内容
        saveInTxtFileByAppend_txt(fileName, htmlContent_str)
        print(htmlContentTitle_str, 'downloaded successful')
        #获取下一个页面链接
        nextHtmlUrl_str = getNextUrl_str(htmlContent_soup, urlPre, nextPageClassName)
        time.sleep(random.randint(2, 4))  #设置程序随机时延，防止断连接
    print('all downloaded successful')