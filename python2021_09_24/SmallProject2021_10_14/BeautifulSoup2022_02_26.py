'''
Author: xudawu
Date: 2022-02-26 16:11:30
LastEditors: xudawu
LastEditTime: 2022-02-26 16:11:31
'''
from bs4 import BeautifulSoup

html = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title" name="dromouse"><b>p标签内容</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1"><!-- Elsie --></a>,
<a href="http://example.com/lacie" class="sister" id="link2">第2个a标签内容</a> and
<a href="http://example.com/tillie" class="sister" id="link3">第3个a标签内容</a>;
and they lived at the bottom of a well.</p>
<div class="page_chapter">
	<ul>
		<li><a href="/html/59000/59000330/index.html">上一章</a></li>
        <li><a href="/html/59000/59000330/index.html">返回目录</a></li>
        <li><a href="/html/59000/59000330/10079982.html">下一章</a></li>
        </li>
	</ul>
</div>
<p class="story">...</p>
"""
soup = BeautifulSoup(html)
#1.显示网页所有内容
print('1', soup)
print('- ' * 10, '分割线', '- ' * 10)
#格式化输出网页所有内容
# print(soup.prettify())
#2.获取html中tag内容
print('2', soup.title)
print('- ' * 10, '分割线', '- ' * 10)
print('2', soup.head)
print('- ' * 10, '分割线', '- ' * 10)
print('2', soup.a)
print('- ' * 10, '分割线', '- ' * 10)
#3.只获取p标签内内容
print('3', soup.p.string)
print('- ' * 10, '分割线', '- ' * 10)
#4.find_all()方法搜索当前 tag 的所有 tag 子节点,返回列表类型
print('4', soup.find_all('p'))
print('- ' * 10, '分割线', '- ' * 10)
print('4', soup.find_all('a'))
print('- ' * 10, '分割线', '- ' * 10)
#5.指定id查找
print('5', soup.find_all(id='link2'))
print('- ' * 10, '分割线', '- ' * 10)
#6.指定class查找
print('6', soup.find_all(class_='sister'))
print('- ' * 10, '分割线', '- ' * 10)
#7.find()与find_all()的区别是，find()直接返回结果，find_all()返回列表
print(soup.find(id='link2'))
print('- ' * 10, '分割线', '- ' * 10)
#8.获取标签内标签内容
print('8', soup.find(class_='title').b)
print('- ' * 10, '分割线', '- ' * 10)
#9.只获取文本内容
print('9', soup.find(class_='story').get_text())
print('- ' * 10, '分割线', '- ' * 10)
#10获取兄弟结点
print('10', soup.find(class_='page_chapter').ul)
print('- ' * 10, '分割线', '- ' * 10)
#11.next_sibling 属性获取了该节点的下一个兄弟节点,下一个兄弟节点是空白或者换行符时继续找下一个兄弟结点
print('11', soup.find(class_='page_chapter').ul.li)
print(soup.find(class_='page_chapter').ul.li.next_sibling)
print(soup.find(class_='page_chapter').ul.li.next_sibling.next_sibling)
print(
    soup.find(class_='page_chapter').ul.li.next_sibling.next_sibling.
    next_sibling.next_sibling)
print(soup.find(class_='page_chapter').ul.li.next_sibling.next_sibling.next_sibling.next_sibling.a.get('href'))
print('- ' * 10, '分割线', '- ' * 10)
