'''
Author: xudawu
Date: 2022-04-06 15:22:15
LastEditors: xudawu
LastEditTime: 2022-04-06 21:55:05
使用 Selenium 从“360图片”网站搜索和下载图片。
'''
# Selenium 是一个自动化测试工具，利用它可以驱动浏览器执行特定的行为，最终帮助爬虫开发者获取到网页的动态内容。
# selenium 库是模拟人工操作浏览器的，优点可见即可爬
import random
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys # Keys 类提供键盘按键的支持
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
    options.add_argument('--headless')
    # 创建Chrome浏览器对象并传入参数
    browser_WebDriver = webdriver.Chrome(options=options)
    # 执行Chrome开发者协议命令（在加载页面时执行指定的JavaScript代码）
    browser_WebDriver.execute_cdp_cmd(
        'Page.addScriptToEvaluateOnNewDocument',
        {'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'}
    )
    browser_WebDriver.get(url)
    return browser_WebDriver

# 存储文件
def saveFile(filePath,fileContent):
    # wb以二进制格式打开一个文件只用于写入
    openFile=open(filePath,'wb')
    openFile.write(fileContent)
    openFile.close()
    print('file:',filePath,'saved')

# 找到所有标签
def findAllTag(tagName_str,name_str):
    allTag=browser_WebDriver.find_elements(tagName_str,name_str)
    return allTag

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
    url='https://image.so.com/c?ch=beauty'
    # 加载指定 URL 的页面到浏览器中
    browser_WebDriver=getBrowser(url)
    # 窗口最大化
    # browser_WebDriver.maximize_window()
    # 当前标签页浏览器渲染之后的网页源代码,包括动态内容
    # html=browser_WebDriver.page_source
    # print(html)
    # 设置隐式等待时间
    # browser_WebDriver.implicitly_wait(5)
    # 显示页面标题
    title_str=browser_WebDriver.title
    print('page:',title_str,'loaded')

    # 通过name获取元素
    searchBoxName_str="q"
    searchElement_input = browser_WebDriver.find_element('name',searchBoxName_str)
    # 先清除预先输入的内容
    searchElement_input.clear()
    # 模拟用户输入行为
    searchElement_input.send_keys('李沁')
    # 输入enter键
    searchElement_input.send_keys(Keys.ENTER)

    #等待页面加载完成
    # time.sleep(2)
    title_str=browser_WebDriver.title
    print('page:',title_str,'loaded')
    
    #瀑布流网页,先翻页加载完毕再下载
    for i in range(5):
        # 直接滚动到页面底部
        browser_WebDriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # 等待加载
        time.sleep(1)

    # 翻多少页
    for i in range(1):
        # 找到所有的图片标签,存储每页数据
        allImgTag=findAllTag('class name','img')
        # 找到所有标签页的子标签
        fileIndex_int=0
        for img in allImgTag:
            # 获取图片src
            imgSrc_str=img.find_element('tag name','img').get_attribute('src')
            # print(imgSrc_str)
            # 获取网络资源
            imgFile_content=getNetFile(imgSrc_str)
            # 存储路径
            fileRoot_Path='img\\'
            fileIndex_int+=1
            # 文件名
            filePath_path=fileRoot_Path+title_str+str(i+1)+'_'+str(fileIndex_int)+'.jpg'
            # 存储文件
            saveFile(filePath_path,imgFile_content)
        #寻找下一页
        time.sleep(random.randint(2, 4))  #随机休眠，避免爬取页面过于频繁,防止被封IP
    # 关闭标签页
    browser_WebDriver.close()
    # 关闭浏览器
    browser_WebDriver.quit()
    print('all download successfully')

# 向下滚动页面
# 直接滚动到页面底部
# browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# 按像素滚动
# browser.execute_script("window.scrollTo(0, 500);")

# 移动到元素element对象的“底端”与当前窗口的“底部”对齐
# browser.execute_script("arguments[0].scrolllntoView(false);", element)

# 滚动至元素可见
# browser.execute_script('window.scrollBy()')
# browser.execute_script('arguments[0].scrollIntoView();', element) # 滚动至元素element可见

# 控制滚动条逐步滚动
# for i in range(20):
#     browser.execute_script('window.scrollBy(0,100)')
#     time.sleep(1)