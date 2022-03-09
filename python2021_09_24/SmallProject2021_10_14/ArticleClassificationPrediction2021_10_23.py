'''
Author: xudawu
Date: 2021-10-23 10:57:59
LastEditors: xudawu
LastEditTime: 2021-10-30 15:37:33
'''

import jieba  #分词器

contents_list='大武喜欢看电影，一鸣也喜欢看电影,一鸣还喜欢玩英雄联盟手游'
words_list=jieba.lcut(str(contents_list))#返回分词列表
print(words_list)

print('- ' * 10, '分割线', '- ' * 10)

vocab_list = list(set(words_list))  #词语去重并返回列表
print(vocab_list)