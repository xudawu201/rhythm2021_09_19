'''
Author: xudawu
Date: 2021-10-21 20:18:37
LastEditors: xudawu
LastEditTime: 2022-03-09 09:16:29
'''
import jieba

#1.jieba的三种基本用法

# 全匹配
seg_list = jieba.cut("今天哪里都没去，在家里睡了一天", cut_all=True)
print(list(seg_list))  # ['今天', '哪里', '都', '没去', '', '', '在家', '家里', '睡', '了', '一天']

# 精确匹配 默认模式
seg_list = jieba.cut("今天哪里都没去，在家里睡了一天", cut_all=False)
print(list(seg_list))  # ['今天', '哪里', '都', '没', '去', '，', '在', '家里', '睡', '了', '一天']

# 精确匹配
seg_list = jieba.cut_for_search("今天哪里都没去，在家里睡了一天")
print(list(seg_list))  # ['今天', '哪里', '都', '没', '去', '，', '在', '家里', '睡', '了', '一天']

print('- '*10,'分割线','- '*10)


#2. 自定义词典
test_sent = """数学是一门基础性的大学课程，深度学习是基于数学的，尤其是线性代数课程"""

words = jieba.cut(test_sent)
print(list(words))
# ['\n', '数学', '是', '一门', '基础性', '的', '大学', '课程', '，', '深度',
# '学习', '是', '基于', '数学', '的', '，', '尤其', '是', '线性代数', '课程', '\n']

words = jieba.cut(test_sent, cut_all=True)
print(list(words))
# ['\n', '数学', '是', '一门', '基础', '基础性', '的', '大学', '课程', '', '', '深度',
# '学习', '是', '基于', '数学', '的', '', '', '尤其', '是', '线性', '线性代数', '代数', '课程', '\n']

jieba.add_word("尤其是")
jieba.add_word("线性代数课程")

words = jieba.cut(test_sent)
print(list(words))
# ['\n', '数学', '是', '一门', '基础性', '的', '大学课程', '，', '深度学习', '是',
# '基于', '数学', '的', '，', '尤其是', '线性代数课程', '\n']

print('- ' * 10, '分割线', '- ' * 10)

print('- ' * 10, '分割线', '- ' * 10)

#4.词性标注
import jieba.posseg as pseg

# 默认模式
seg_list = pseg.cut("今天哪里都没去，在家里睡了一天")
for word, flag in seg_list:
    print(word + " " + flag)
"""
使用 jieba 默认模式的输出结果是：
今天 t
哪里 r
都 d
没去 v
， x
在 p
家里 s
睡 v
了 ul
一天 m
"""

print('- ' * 10, '分割线', '- ' * 10)

# paddle 模式
words = pseg.cut("今天哪里都没去，在家里睡了一天", use_paddle=True)
for word, flag in words:
    print(word + " " + flag)
"""
使用 paddle 模式的输出结果是：
今天 t
哪里 r
都 d
没去 v
， x
在 p
家里 s
睡 v
了 ul
一天 m
"""
print('- ' * 10, '分割线', '- ' * 10)

#5.获取词语位置
result = jieba.tokenize('华为技术有限公司的手机品牌')
for tk in result:
    print("word:" + tk[0] + " start:" + str(tk[1]) + " end:" + str(tk[2]))
"""
word:华为技术有限公司 start:0 end:8
word:的 start:8 end:9
word:手机 start:9 end:11
word:品牌 start:11 end:13
"""

print('- ' * 10, '分割线', '- ' * 10)

# 使用 search 模式
result = jieba.tokenize('华为技术有限公司的手机品牌', mode="search")
for tk in result:
    print("word:" + tk[0] + " start:" + str(tk[1]) + " end:" + str(tk[2]))

print('- ' * 10, '分割线', '- ' * 10)

#6.收索引擎
# 使用 jieba 和 whoosh 可以实现搜索引擎功能。whoosh 是由python实现的一款全文搜索工具包
import os
import shutil

from whoosh.fields import *
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from jieba.analyse import ChineseAnalyzer

analyzer = ChineseAnalyzer()

schema = Schema(title=TEXT(stored=True),
                path=ID(stored=True),
                content=TEXT(stored=True, analyzer=analyzer))
if not os.path.exists("test"):
    os.mkdir("test")
else:
    # 递归删除目录
    shutil.rmtree("test")
    os.mkdir("test")

idx = create_in("test", schema)
writer = idx.writer()

#添加中文
writer.add_document(title=u"document1",
                    path="/tmp1",
                    content=u"麦迪是一位著名的篮球运动员，他飘逸的打法深深吸引着我")
writer.add_document(title=u"document2",
                    path="/tmp2",
                    content=u"科比是一位著名的篮球运动员，他坚韧的精神深深的感染着我")
writer.add_document(title=u"document3", path="/tmp3", content=u"詹姆斯是我不喜欢的运动员")

writer.commit()
searcher = idx.searcher()
parser = QueryParser("content", schema=idx.schema)

#执行搜索
for keyword in ("篮球", "麦迪"):
    print("searched keyword ", keyword)
    query = parser.parse(keyword)
    results = searcher.search(query)
    for hit in results:
        print(hit.highlights("content"))
    print("=" * 50)
'''
searched keyword  篮球
麦迪是一位著名的<b class="match term0">篮球</b>运动员，他飘逸的打法深深吸引着我
科比是一位著名的<b class="match term0">篮球</b>运动员，他坚韧的精神深深的感染着我
==================================================
searched keyword  麦迪
<b class="match term0">麦迪</b>是一位著名的篮球运动员，他飘逸的打法深深吸引着我
'''

"""
输出：
word:华为 start:0 end:2
word:技术 start:2 end:4
word:有限 start:4 end:6
word:公司 start:6 end:8
word:华为技术有限公司 start:0 end:8
word:的 start:8 end:9
word:手机 start:9 end:11
word:品牌 start:11 end:13
"""


'''
1.jieba常用的三种模式：

精确模式，试图将句子最精确地切开，适合文本分析；
全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
可使用 jieba.cut 和 jieba.cut_for_search 方法进行分词，
两者所返回的结构都是一个可迭代的 generator，可使用 for 循环来获得分词后得到的每一个词语（unicode），
或者直接使用 jieba.lcut 以及 jieba.lcut_for_search 返回 list。

jieba.Tokenizer(dictionary=DEFAULT_DICT) ：使用该方法可以自定义分词器，
可以同时使用不同的词典。jieba.dt 为默认分词器，所有全局分词相关函数都是该分词器的映射。

jieba.cut 和 jieba.lcut 可接受的参数如下：

需要分词的字符串（unicode 或 UTF-8 字符串、GBK 字符串）
cut_all：是否使用全模式，默认值为 False
HMM：用来控制是否使用 HMM 模型，默认值为 True
jieba.cut_for_search 和 jieba.lcut_for_search 接受 2 个参数：

需要分词的字符串（unicode 或 UTF-8 字符串、GBK 字符串）
HMM：用来控制是否使用 HMM 模型，默认值为 True
需要注意的是，尽量不要使用 GBK 字符串，可能无法预料地错误解码成 UTF-8。
'''

'''
2.自定义词典

开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词。
用法： jieba.load_userdict(dict_path)
dict_path：为自定义词典文件的路径
词典格式如下：
一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。
jieba.add_word()：向自定义字典中添加词语
'''

'''
3.关键词提取

可以基于TF-IDF 算法进行关键词提取，也可以基于extRank 算法。 TF-IDF 算法与 elasticsearch 中使用的算法是一样的。
使用 jieba.analyse.extract_tags() 函数进行关键词提取，其参数如下：

jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())

sentence 为待提取的文本
topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
withWeight 为是否一并返回关键词权重值，默认值为 False
allowPOS 仅包括指定词性的词，默认值为空，即不筛选
jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
也可以使用 jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件。
'''

'''
4.词性标注
词性标注主要是标记文本分词后每个词的词性，
paddle模式的词性
'''

'''
5.获取词语位置
将分本分词后，返回每个词和该词在原文中的起始位置，
'''

'''
6.收索引擎

使用 jieba 和 whoosh 可以实现搜索引擎功能。
whoosh 是由python实现的一款全文搜索工具包，可以使用 pip 安装它
'''