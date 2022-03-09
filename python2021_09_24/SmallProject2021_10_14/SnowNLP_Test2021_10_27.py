'''
Author: xudawu
Date: 2021-10-27 15:57:31
LastEditors: xudawu
LastEditTime: 2021-10-27 17:19:49
'''
# -*- coding: utf-8 -*-
from snownlp import SnowNLP
import snownlp

s = SnowNLP(u"这个东西真心很赞")

print("1、中文分词:\n", s.words)

print("2、词性标注:\n", s.tags)

print("3、情感倾向分数越接近1越正向:\n", s.sentiments)

print("4、转换拼音:\n", s.pinyin)

print("5、输出前4个关键词:\n", s.keywords(4))

print("6、概括文章文意:\n", s.summary(1))

print("7.1、输出tf:\n", s.tf)

print("7.2、输出idf:\n", s.idf)

text1=u'''自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
所以它与语言学的研究有着密切的联系，但又有重要的区别。
自然语言处理并不是一般地研究自然语言，
而在于研制能有效地实现自然语言通信的计算机系统，
特别是其中的软件系统。因而它是计算机科学的一部分。'''

text1_snowNLP = SnowNLP(text1)

print('- ' * 10, '分割线', '- ' * 10)
print(text1_snowNLP.sentiments)

print('- ' * 10, '分割线', '- ' * 10)
print(text1_snowNLP.summary(2))

print('- ' * 10, '分割线', '- ' * 10)
print(text1_snowNLP.keywords(6))

print('- ' * 10, '分割线', '- ' * 10)
#文本相似
s = SnowNLP([['语言', '自然'],
             ['领域', '计算机'],
             ['电脑']])

print(s.sim(['计算机']))