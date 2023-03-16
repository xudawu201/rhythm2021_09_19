'''
Author: xudawu
Date: 2023-03-15 09:02:59
LastEditors: xudawu
LastEditTime: 2023-03-16 11:59:10
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
import os
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
    # 禁用“保存密码”弹出窗口
    options.add_experimental_option("prefs",{"credentials_enable_service":False,"profile.password_manager_enabled":False})

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
    WebDriverWait(browser_WebDriver,20).until(EC.presence_of_element_located(pageLocator_str))

# 找到页面元素并点击
def findElementAndClick(elementType_str,elementName_str):
    findElement_WebDriver = browser_WebDriver.find_element(elementType_str,elementName_str)
    findElement_WebDriver.click()

# 判断是否存在某元素
def isElemenntExit(elementType_str,elementName_str):
    flag = True
    try:
        browser_WebDriver.find_element(elementType_str,elementName_str)
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

# 用户登录
def loginIn(userName_str,password_str):
    # 账号
    loginNameElementType_str='name'
    loginNameElementName_str='userName'
    findloginNameElement_WebDriver=browser_WebDriver.find_element(loginNameElementType_str,loginNameElementName_str)
    findloginNameElement_WebDriver.send_keys(userName_str)

    # 密码
    loginPasswordElementType_str='name'
    loginPasswordElementName_str='password'
    findloginPasswordElement_WebDriver=browser_WebDriver.find_element(loginPasswordElementType_str,loginPasswordElementName_str)
    findloginPasswordElement_WebDriver.send_keys(password_str)

    # 点击登录
    loginElementType_str='class name'
    loginElementName_str='j-login'
    findloginElement_WebDriver=browser_WebDriver.find_element(loginElementType_str,loginElementName_str)
    findloginElement_WebDriver.send_keys(Keys.ENTER)

# 获取txt新闻内容
def getTxtContent(fileName_str):
    
    # 文件名
    fileName_reader=fileName_str
    # 读取文件
    file_reader = open(fileName_reader, mode='r', encoding='utf-8')
    # 新闻数
    newCount_int=0
    # 新闻标识符
    newTag_str='新闻标识符'
    # 存储新闻的二维列表
    news=[[]]
    # 逐行读取
    isNextNew_int=0
    for line in file_reader.readlines():
        # 读到了一条新闻
        if newTag_str in line : 
            isNextNew_int=isNextNew_int+1      
            # 是下一条新闻
            if isNextNew_int==2:
                # 新闻数加一
                news.append([])
                newCount_int=newCount_int+1
                isNextNew_int=1
            # 扩展列表
            # 去掉标识符号
            tempNewTag=line
            newContent_str=tempNewTag.replace('新闻标识符','').replace('\n', '')
            news[newCount_int].append(newContent_str)
            continue
        # 遇到空行直接跳过
        if line=='\n':
            continue
        tempNewTag=line
        # 去掉换行符号
        temp2NewTag=tempNewTag.replace('\n', '')
        news[newCount_int].append(temp2NewTag)
    
    # 关闭文件
    file_reader.close()
    return news


# 获取txt国家内容
def getCountryInfo(fileName_str):
    
    # 文件名
    fileName_reader=fileName_str
    # 读取文件
    file_reader = open(fileName_reader, mode='r', encoding='utf-8')
    # 国家名
    country_str=''
    # 国家发布新闻机构
    countryAgency_str=''
    # 存储国家信息列表
    countryInfo=[]
    # 逐行读取
    for line in file_reader.readlines():
        if line=='\n':
            continue
        tempInfo=line
        # 去掉换行符号
        tempInfo2=tempInfo.replace('\n', '')
        countryInfo.append(tempInfo2)
    
    # 关闭文件
    file_reader.close()
    return countryInfo

# 点击国家热点信息
def clickHotInf(country_str,columnAddDiv_WebDriver):

    attempts = 0
    while attempts < 2:
        try:
            # 点击下拉框
            aTag_WebDriver=columnAddDiv_WebDriver.find_elements('tag name','input')[0]
            aTag_WebDriver.click()
            break
        except StaleElementReferenceException:
            attempts += 1
    # 定义一个列表包含10个国家名字
    countries = ['缅甸', '泰国', '越南', '柬埔寨', '老挝', '印度', '孟加拉', '尼泊尔', '巴基斯坦', '斯里兰卡']

    # 获取用户输入的国家名字
    thisCountry_name = country_str
    attempts = 0
    while attempts < 2:
        try:
            # 判断国家名字是否在列表中
            if thisCountry_name in countries:
                # 如果在列表中，输出该国家在列表中的位置（从0开始）
                # 返回是哪一个国家
                index=countries.index(thisCountry_name)
                # 是缅甸
                if index==0:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_269')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 泰国
                if index==1:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_281')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 越南
                if index==2:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_293')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 柬埔寨
                if index==3:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_305')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 老挝
                if index==4:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_317')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 印度
                if index==5:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_329')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 孟加拉
                if index==6:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_341')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 尼泊尔
                if index==7:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_353')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 巴基斯坦
                if index==8:
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_365')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
                # 斯里兰卡
                if index==9:
                    time.sleep(1)
                    hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_377')
                    browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                    break
            else:
                # time.sleep(1)
                # 如果不在列表中,直接热点信息
                hotInfo_WebDriver=browser_WebDriver.find_element('id','_easyui_tree_261')
                # hotInfo_WebDriver.click()
                # 使用js实现点击
                browser_WebDriver.execute_script("arguments[0].click();", hotInfo_WebDriver)
                break
        except StaleElementReferenceException:
            attempts += 1
    
# 控制页面提交新闻信息
def controlPage(country_str,cuntryAgency_str,newIndex_int,newTitle_str,newUrl_str,contentList_list):

    # 等待页面加载完成
    pageLocator_str=('id','toolbar')
    waitPageLoaded(pageLocator_str)

    # 找到工具栏
    toolDivElementType_str='id'
    toolDivElementName_str='toolbar'
    toolDivElement_WebDriver=browser_WebDriver.find_element(toolDivElementType_str,toolDivElementName_str)

    # 找到添加按钮并点击
    addNewElementType_str='link text'
    addNewElementName_str='添加'
    addNewElement_WebDriver=toolDivElement_WebDriver.find_element(addNewElementType_str,addNewElementName_str)
    # 直接点击可能会被页面挡住，直接模拟输入enter键
    addNewElement_WebDriver.send_keys(Keys.ENTER)

    # 等待页面加载完成
    pageLocator_str=('id','crud_ew-buttons')
    waitPageLoaded(pageLocator_str)

    # 找到新闻输入列表
    tableElementType_str='class name'
    tableElementName_str='edit_form_table'
    tableElement_WebDriver=browser_WebDriver.find_element(tableElementType_str,tableElementName_str)
    
    # 找到新闻标题输入框
    newTitleElementType_str='tag name'
    newTitleElementName_str='tr'
    newTitleElement_WebDriver=tableElement_WebDriver.find_elements(newTitleElementType_str,newTitleElementName_str)[0]
    # 点击一下确定选中输入框
    # newTitleElement_WebDriver.click()
    # 定位新闻标题输入框
    inputTitleElement_WebDriver=newTitleElement_WebDriver.find_elements('tag name','input')[1]
    # 输入新闻标题
    inputTitleElement_WebDriver.send_keys(newTitle_str)

    # 定位国家输入框
    # 找到新闻标题输入框
    countryElement_WebDriver=tableElement_WebDriver.find_elements('tag name','tr')[5]
    inputTitleElement_WebDriver=countryElement_WebDriver.find_elements('tag name','input')[1]
    # 输入新闻标题
    inputTitleElement_WebDriver.send_keys(country_str)
    
    # 定位工具栏
    textToolDivElement_WebDriver=browser_WebDriver.find_element('class name','ke-toolbar')

    # 点击超链接图标
    newUrlButton_WebDriver=textToolDivElement_WebDriver.find_elements('tag name','span')[102]
    newUrlButton_WebDriver.click()

    # 等待加载完成
    pageLocator_str=('class name','ke-dialog-footer')
    waitPageLoaded(pageLocator_str)    

    # 输入超链接
    newUrl_WebDriver=browser_WebDriver.find_element('id','keUrl')
    # 清除原先预输入
    newUrl_WebDriver.send_keys(Keys.CONTROL, 'a')
    newUrl_WebDriver.send_keys(newUrl_str)

    # 点击确定
    confrimMainElement_WebDriver=browser_WebDriver.find_element('class name','ke-dialog-footer')
    confrimElement_WebDriver=confrimMainElement_WebDriver.find_element('tag name','input')
    confrimElement_WebDriver.send_keys(Keys.ENTER)

    browser_WebDriver.implicitly_wait(1)

    # 输入文章正文
    # 找到窗口
    browser_WebDriver.switch_to.frame(0)

    # 输入文章正文
    newContent_WebDriver=browser_WebDriver.find_element('class name','ke-content')
    # 倒着输出文本
    for contentIndex_int in range(len(contentList_list)-1,-1,-1):
        # 回车换行
        newContent_WebDriver.send_keys(Keys.ENTER)
        newContent_WebDriver.send_keys(contentList_list[contentIndex_int])

    # 回到上层窗口
    browser_WebDriver.switch_to.parent_frame()
    # 上传新闻图片
    thisNewCount_int=newIndex_int
    newsRootPath_str='news'
    newsFileList=os.listdir(newsRootPath_str)
    # print(newsFileList)
    for newName in newsFileList:
        # 如果有图片则上传
        if newName==str(thisNewCount_int)+'.png':
            # 点击上传图片图标
            newUrlButton_WebDriver=textToolDivElement_WebDriver.find_elements('tag name','span')[80]
            newUrlButton_WebDriver.click()
            # 等待加载完成
            pageLocator_str=('class name','ke-tabs')
            waitPageLoaded(pageLocator_str) 
            # 点击本地上传
            localUploadDivElement_WebDriver=browser_WebDriver.find_element('class name','ke-tabs')
            localUploadElement_WebDriver=localUploadDivElement_WebDriver.find_elements('tag name','li')[1]
            # 点击
            localUploadElement_WebDriver.click()
            # 点击浏览
            # 获取图片绝对路径
            newRealName_str=os.path.join(newsRootPath_str,newName)
            newAbsName_str = os.path.abspath(newRealName_str)
            # 上传
            searchLocalElement_WebDriver=browser_WebDriver.find_element('name','imgFile')
            searchLocalElement_WebDriver.send_keys(newAbsName_str)
            # searchLocalElement_WebDriver.click()
            
            # 点击确定上传
            confirmButtonElement_WebDriver=browser_WebDriver.find_element('class name','ke-dialog-footer')
            confirmButtonElement_WebDriver.find_element('tag name','input').click()
            # 等待加载完成
            time.sleep(3)
            # pageLocator_str=('class name','ke-toolbar')
            # waitPageLoaded(pageLocator_str)
           
            # 上传完成退出循环
            break

    attempts = 0
    while attempts < 2:
        try:
            # 点击去除html标记图标
            removeHtmlButton_WebDriver=textToolDivElement_WebDriver.find_elements('tag name','span')[50]
            # removeHtmlButton_WebDriver.click()
            browser_WebDriver.execute_script("arguments[0].click();", removeHtmlButton_WebDriver)
        
            # 点击自动排版图标
            autoLayoutButton_WebDriver=textToolDivElement_WebDriver.find_elements('tag name','span')[52]
            # autoLayoutButton_WebDriver.click()
            browser_WebDriver.execute_script("arguments[0].click();", autoLayoutButton_WebDriver)
            break
        except StaleElementReferenceException:
            attempts += 1

    # 等待加载完成
    # pageLocator_str=('class name','ke-tabs')
    # waitPageLoaded(pageLocator_str)
    # 定位信息来源div
    attempts = 0
    while attempts < 2:
        try:
            newTableDiv_WebDriver=browser_WebDriver.find_element('id','crud_ef')
            # table是以1开头不是0开头
            newFromDiv_WebDriver=newTableDiv_WebDriver.find_elements('tag name','table')[2]
            # 定位input
            newFromInput_WebDriver=newFromDiv_WebDriver.find_elements('tag name','input')[1]
            newFromInput_WebDriver.send_keys(countryAgency_str)
            break
        except StaleElementReferenceException:
            attempts += 1
    
    attempts = 0
    while attempts < 2:
        try:
            # 定位栏目div
            columnDiv_WebDriver=browser_WebDriver.find_element('id','category_edit_form')
            columnAddDiv_WebDriver=columnDiv_WebDriver.find_elements('tag name','tr')[3]
            # 点击添加栏目
            columnAddDiv_WebDriver.click()
            break
        except StaleElementReferenceException:
            attempts += 1
    

    # 等待加载完成
    pageLocator_str=('class name','textbox-addon')
    waitPageLoaded(pageLocator_str)

    # 点击热点信息
    clickHotInf(country_str,columnAddDiv_WebDriver)
    
    attempts = 0
    while attempts < 2:
        try:
            # 点击保存
            confirmDiv_WebDriver=browser_WebDriver.find_element('id','crud_ew-buttons')
            confirmButton_WebDriver=confirmDiv_WebDriver.find_element('tag name','a')
            confirmButton_WebDriver.send_keys(Keys.ENTER)

            # 点击确定弹窗
            confirmDiv_WebDriver=browser_WebDriver.find_element('class name','messager-button')
            confirmButton_WebDriver=confirmDiv_WebDriver.find_element('tag name','a')
            confirmButton_WebDriver.send_keys(Keys.ENTER)
            break
        except StaleElementReferenceException:
            attempts += 1       
    
    # 等待页面加载完成
    pageLocator_str=('id','toolbar')
    waitPageLoaded(pageLocator_str)

    # 找到3条最近上传的新闻判断哪一条是刚才上传的进行点击选中
    attempts = 0
    while attempts < 2:
        try:
            table =browser_WebDriver.find_element('id','datagrid-row-r1-2-0')
            thisTitle=table.find_element('tag name','td').text
            if thisTitle==newTitle_str:
                table.click()
                break
            table =browser_WebDriver.find_element('id','datagrid-row-r1-2-1')
            thisTitle=table.find_element('tag name','td').text
            if thisTitle==newTitle_str:
                table.click()
                break
            table =browser_WebDriver.find_element('id','datagrid-row-r1-2-2')
            thisTitle=table.find_element('tag name','td').text
            if thisTitle==newTitle_str:
                table.click()
                break
        except StaleElementReferenceException:
            attempts += 1

    # 找到添加按钮并点击
    attempts = 0
    while attempts < 2:
        try:
            # 找到添加按钮并点击
            pubulishElement_WebDriver=toolDivElement_WebDriver.find_element('link text','发布')
            # 直接点击可能会被页面挡住，直接模拟输入enter键
            pubulishElement_WebDriver.click()
            break
        except StaleElementReferenceException:
            attempts += 1
            
    # 点击确定弹窗
    attempts = 0
    while attempts < 2:
        try:
            confirmDiv_WebDriver=browser_WebDriver.find_element('class name','messager-button')
            confirmButton_WebDriver=confirmDiv_WebDriver.find_element('tag name','a')
            confirmButton_WebDriver.click()
            break
        except StaleElementReferenceException:
            attempts += 1

    # 再次点击确定弹窗
    attempts = 0
    while attempts < 2:
        try:
            confirmDiv_WebDriver=browser_WebDriver.find_element('class name','messager-button')
            confirmButton_WebDriver=confirmDiv_WebDriver.find_element('tag name','a')
            confirmButton_WebDriver.click()
            break
        except StaleElementReferenceException:
            attempts += 1

#主函数
if __name__ == '__main__':
    # 加载指定的页面
    # 云南省外事办http://218.78.91.39:8081/auth/login
    url='http://218.78.91.39:8081/auth/login'
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
    pageLocator_str=('class name','el-form')
    waitPageLoaded(pageLocator_str)
    
    # 登录账号
    loginIn('wsbadmin', '1')

    # 等待登录完成
    # 设置隐式等待时间
    # browser_WebDriver.implicitly_wait(5)
    # html=browser_WebDriver.page_source
    # print(html)

    # 等待页面加载完成
    pageLocator_str=('id','menu')
    waitPageLoaded(pageLocator_str)

    # 刷新网页
    # browser_WebDriver.refresh()

    # 等待页面加载完成
    # pageLocator_str=('id','menu')
    # waitPageLoaded(pageLocator_str)

    # html=browser_WebDriver.page_source
    # print(html)

    # 信息发布管理
    newPublishElementType_str='link text'
    newPublishElementName_str='信息发布管理'
    newPublishElement_WebDriver=browser_WebDriver.find_element(newPublishElementType_str,newPublishElementName_str)
    newPublishElement_WebDriver.send_keys(Keys.ENTER)

    # 切换iframe
    browser_WebDriver.switch_to.frame('main')
    
    # 获取txt国家配置信息
    countryInfo=getCountryInfo('config.txt')
    country_str=countryInfo[0]
    countryAgency_str=countryInfo[1]
    # 获取txt中新闻信息
    news=getTxtContent('news.txt')
    # 上传新闻
    newIndex_int=0
    newTitle_str=''
    newUrl_str=''
    newContent_str=''
    for contentList_index in range(len(news)):
        newIndex_int=news[contentList_index][0]
        newTitle_str=news[contentList_index][2]
        newUrl_str=news[contentList_index][3]
        contentList_list=[]
        for index in range(4,len(news[contentList_index])):
            # newContent_str=news[contentList_index][index]
            contentList_list.append(news[contentList_index][index])
            # print(newContent_str)
        controlPage(country_str,countryAgency_str,newIndex_int,newTitle_str,newUrl_str,contentList_list)
        # 每上传一次新闻暂停
        time.sleep(3)
    
    # 关闭浏览器
    browser_WebDriver.quit()
    print('all save successfully')