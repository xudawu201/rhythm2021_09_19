'''
Author: xudawu
Date: 2023-03-16 13:47:16
LastEditors: xudawu
LastEditTime: 2023-03-23 09:35:11
'''
# Selenium 是一个自动化测试工具，利用它可以驱动浏览器执行特定的行为，最终帮助爬虫开发者获取到网页的动态内容。
# selenium 库是模拟人工操作浏览器的，优点可见即可爬
import random
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys # Keys 类提供键盘按键的支持
from selenium.webdriver.support.wait import WebDriverWait # 设置显示等待
from selenium.webdriver.support import expected_conditions as EC # 设置显示等待，找到特定标签后开始操作
from selenium.webdriver.support.ui import Select # 选择下拉框
from selenium.common.exceptions import StaleElementReferenceException
'''
1.有一些网站专门针对 Selenium 设置了反爬措施，因为使用 Selenium 驱动的浏览器，
在控制台中可以看到如下所示的webdriver属性值为true，如果要绕过这项检查，
可以在加载页面之前，先通过执行 JavaScript 代码将其修改为undefined。
2.将浏览器窗口上的“Chrome正受到自动测试软件的控制”隐藏掉
'''

# 获得浏览器对象
def getBrowser(url):
    # 创建Chrome参数对象
    options = webdriver.ChromeOptions()
    # 添加试验性参数
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    # 不需要显示浏览器窗口
    # options.add_argument('--headless')
    # 创建Chrome浏览器对象并传入参数
    browser_WebDriver = webdriver.Chrome(options=options)
    # 执行Chrome开发者协议命令（在加载页面时执行指定的JavaScript代码）
    browser_WebDriver.execute_cdp_cmd(
        'Page.addScriptToEvaluateOnNewDocument',
        {'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'}
    )
    browser_WebDriver.get(url)
    return browser_WebDriver

# 等待页面加载完成
def waitPageLoaded(pageLocator_str):
    WebDriverWait(browser_WebDriver,30).until(EC.presence_of_element_located(pageLocator_str))

# 找到页面元素并点击
def findElementAndClick(elementType_str,elementName_str):
    findElement_WebDriver = browser_WebDriver.find_element(elementType_str,elementName_str)
    findElement_WebDriver.click()

# 判断是否存在某元素
def isElemenntExit(elementType_str,elementName_str):
    flag = True
    try:
        img_WebDriver=browser_WebDriver.find_element(elementType_str,elementName_str)
        img_WebDriver.find_element('tag name','img')
        return flag
    except:
        flag = False
        return flag

# 存储为txt文件，以追加方式
def saveInTxtFileByAppend_txt(fileName, content_str):
    file_txt = open(fileName, mode='a', encoding='utf-8')
    file_txt.write(content_str)  # write 写入
    file_txt.write('\r\n') # 写完一次换行
    file_txt.close()  # 关闭文件

# 存储图片文件
def saveImgFile(filePath,fileContent):
    # wb以二进制格式打开一个文件只用于写入
    openFile=open(filePath,'wb')
    openFile.write(fileContent)
    openFile.close()
    print('file:',filePath,'saved')

# 获取网络资源
def getNetFile(url_str):
    # 头部伪装
    # 如果不设置HTTP请求头中的User-Agent，网页会检测出不是浏览器而阻止我们的请求。
    # 通过get函数的headers参数设置User-Agent的值，具体的值可以在浏览器的开发者工具查看到。
    # 用爬虫访问大部分网站时，将爬虫伪装成来自浏览器的请求都是非常重要的一步。
    headers={
        'User-Agent': 
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
        }
    #请求服务器,获取网络资源
    response = requests.get(url_str, headers=headers)
    # 获取网络资源的内容
    content=response.content
    return content

#主函数
if __name__ == '__main__':
    # 加载指定的页面
    # 沙特阿拉伯https://www.spa.gov.sa/newsheadlines.php?lang=ar#page=1
    url='https://www.spa.gov.sa/newsheadlines.php?lang=ar#page=1'
    # 加载指定 URL 的页面到浏览器中
    browser_WebDriver=getBrowser(url)
    # 窗口最大化
    # browser_WebDriver.maximize_window()
    # 当前标签页浏览器渲染之后的网页源代码,包括动态内容
    # html=browser_WebDriver.page_source
    # print(html)
    # 设置隐式等待时间
    # browser_WebDriver.implicitly_wait(5)

    # 等待页面加载完成
    pageLocator_str=('id','footer')
    waitPageLoaded(pageLocator_str)

    # # 定位话题下拉框元素位置
    # topicElementType_str='id'
    # topicElementName_str='topic'
    # findTopicElement_WebDriver=browser_WebDriver.find_element(topicElementType_str,topicElementName_str)
    # # 选择话题下拉框
    # topicElementSelect_Select=Select(findTopicElement_WebDriver)
    # topicElementSelect_Select.select_by_index(6)

    # # 等待页面加载完成
    # pageLocator_str=('id','footer')
    # waitPageLoaded(pageLocator_str)
    
    # 获取新闻列表
    # 定位新闻列表div
    newsListDivelementType_str='class name'
    newsListDivelementName_str='block_home_post'
    newsListDiv_WebDriver=browser_WebDriver.find_element(newsListDivelementType_str, newsListDivelementName_str)

    # 定位新闻列表
    newsListelementType_str='class name'
    newsListelementName_str='tabcontents_area'
    newsList_WebDriver=newsListDiv_WebDriver.find_elements(newsListelementType_str, newsListelementName_str)
    # # 再找到新闻列表
    # newsList_WebDriver
    # 创建文件夹存储新闻图片
    import os
    fileRoot_Path = 'news\\'
    if os.path.exists(fileRoot_Path):
        pass
    else:
        os.mkdir(fileRoot_Path)

    # 新闻数计数
    newsCount_int=1
    # 循环找到新闻并存储
    for newLi_WebDriver in newsList_WebDriver:
        newDiv_WebDriver=newLi_WebDriver.find_elements('class name','occ_title_div')
        for new_WebDriver in newDiv_WebDriver:
            try:
                # 点击进入新闻详情页
                # 用js在新的标签页打开链接
                aNewLi_WebDriver=new_WebDriver.find_element('tag name','a').get_attribute('href')
                browser_WebDriver.execute_script(f'window.open("{aNewLi_WebDriver}", "_blank");')
                # aNewLi_WebDriver.click()

                # 获取所有标签页句柄 
                windows = browser_WebDriver.window_handles

            except StaleElementReferenceException:
                print('未能打开新标签页')
                break

            # # 如果当前标签页只有1一个,即本页新闻全部提取完毕,退出循环
            # if len(windows)==1:
            #     break

            # 切换到目标标签页
            browser_WebDriver.switch_to.window(windows[1])

            # 等待页面加载完成
            pageLocator_str=('id','footer')
            waitPageLoaded(pageLocator_str)
                    
            # 获取新闻发布时间
            newPublishTime_str=browser_WebDriver.find_element('class name','socal_share').text
            # print(newPublishTime_str)
        
            # 存储新闻计数和发布时间
            fileName='news.txt'
            newMark_str='新闻标识符'+str(newsCount_int)
            saveInTxtFileByAppend_txt(fileName,newMark_str)
            saveInTxtFileByAppend_txt(fileName,'发布时间 '+newPublishTime_str)

            # 获取新闻标题
            newTitle_str=browser_WebDriver.find_element('id','h4NewsTitle').text
            # print(newTitle_str)        
            # 存储新闻标题
            fileName='news.txt'
            saveInTxtFileByAppend_txt(fileName,newTitle_str)
            print(newTitle_str,' saved')

            # 获取新闻原文链接
            newsUrl_str=browser_WebDriver.current_url
            # 存储新闻链接
            fileName='news.txt'
            saveInTxtFileByAppend_txt(fileName,newsUrl_str)

            # 获取新闻文章内容
            newContent_str=browser_WebDriver.find_element('class name','divNewsDetailsText').text
            # 存储新闻文章内容
            fileName='news.txt'
            saveInTxtFileByAppend_txt(fileName,newContent_str)

            # 查看是否有新闻照片元素
            isElemenntExit_bool=isElemenntExit('id','divNewsSlider')
            # 如果有图片
            if isElemenntExit_bool==True:
                # 找到图片标签
                newImgATag_str=browser_WebDriver.find_element('id','divNewsSlider')
                # 存储图片
                imgFileName_str=fileRoot_Path+str(newsCount_int)+'.png'
                newImgUrl_str=newImgATag_str.find_element('tag name','img').get_attribute('src')
                # 获取网络资源
                imgFile_content=getNetFile(newImgUrl_str)
                # 存储文件
                saveImgFile(imgFileName_str,imgFile_content)

            # 新闻计数加一
            newsCount_int=newsCount_int+1

            # 关闭标签页
            browser_WebDriver.close()
            # 切换回最初的标签页
            browser_WebDriver.switch_to.window(windows[0])

            # 等待页面加载完成
            pageLocator_str=('id','footer')
            waitPageLoaded(pageLocator_str)

            # 再等待2秒
            browser_WebDriver.implicitly_wait(2)

    # 关闭浏览器
    browser_WebDriver.quit()
    print('all save successfully')