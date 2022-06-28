#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: #DC143C;"><b style="color:white;">Jieba Library</b></h1></center>
# 
# Jieba (Chinese for “to stutter”) Chinese text segmentation.  
# 
# https://github.com/fxsjy/jieba
# 
# "Jieba library is an excellent chinese word segmentation third-party library, it can a chinese word bank is used to determine the correlation probability between chinese characters." 
# 
# "The phrase with high probability between chinese characters is formed to form the result of word segmentation ， will chinese text obtains individual words through word segmentation."
#  
# Jieba three patterns of participles ： accurate model,  all model, search engine model.
#  
# - accurate model ： cut the text exactly, no redundant words
# 
# - all model ： scan out all possible words in the text. it's redundant
# 
# - search engine model ： on the basis of the exact model, the long words are segmented again
# 
# https://www.codestudyblog.com/cnb11/1124183155.html
# 
# 
# "To stutter means to speak or say something, especially the first part of a word, with difficulty, for example pausing before it or repeating it several times: She stutters a little, so be patient and let her finish what she's saying."
# 
# https://dictionary.cambridge.org/pt/dicionario/ingles/stutter

# In[ ]:


df = pd.read_csv('../input/gamers-negative-chat-recognition/train.csv',delimiter=',', encoding='utf-8')
df.head()


# #分词 - First Step

# In[ ]:


get_ipython().system('pip install jieba')


# In[ ]:


# encoding=utf-8
import jieba

jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
for str in strs:
    seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
    print("Paddle Mode: " + '/'.join(list(seg_list)))

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))


# 【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
# 
# 【精确模式】: 我/ 来到/ 北京/ 清华大学
# 
# 【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)
# 
# 【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造

# #添加自定义词典 - Second Step
# 
# 创新办 3 i
# 
# 云计算 5
# 
# 凱特琳 nz
# 
# 台中

# In[ ]:


print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
#如果/放到/post/中将/出错/。
jieba.suggest_freq(('中', '将'), True)
494
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
#如果/放到/post/中/将/出错/。
print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
#「/台/中/」/正确/应该/不会/被/切开
jieba.suggest_freq('台中', True)
69
print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
#「/台中/」/正确/应该/不会/被/切开


# #关键词提取 - Third Step
# 
# 基于 TF-IDF 算法的关键词抽取 - import jieba.analyse
# 
# jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
# 
# sentence 为待提取的文本
# 
# topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
# 
# withWeight 为是否一并返回关键词权重值，默认值为 False
# 
# allowPOS 仅包括指定词性的词，默认值为空，即不筛选
# 
# jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
# 
# 用法： jieba.analyse.set_stop_words(file_name) # file_name为自定义语料库的路径
# 
# 自定义语料库示例：https://github.com/fxsjy/jieba/blob/master/extra_dict/stop_words.txt
# 用法示例：https://github.com/fxsjy/jieba/blob/master/test/extract_tags_stop_words.py

# #词性标注  - Fourth Step
# 
# jieba.posseg.POSTokenizer(tokenizer=None) 新建自定义分词器，tokenizer 参数可指定内部使用的 
# 
# jieba.Tokenizer 分词器。jieba.posseg.dt 为默认词性标注分词器。
# 
# 标注句子分词后每个词的词性，采用和 ictclas 兼容的标记法。
# 
# 除了jieba默认分词模式，提供paddle模式下的词性标注功能。paddle模式采用延迟加载方式，通过enable_paddle()
# 
# 安装paddlepaddle-tiny，并且import相关代码；
# 
# 用法示例

# In[ ]:


import jieba
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门") #jieba默认模式
jieba.enable_paddle() #启动paddle模式。 0.40版之后开始支持，早期版本不支持
words = pseg.cut("我爱北京天安门",use_paddle=True) #paddle模式
for word, flag in words:
    print('%s %s' % (word, flag))

#我 r
#爱 v
#北京 ns
#天安门 ns


# paddle模式词性标注对应表如下：
# 
# paddle模式词性和专名类别标签集合如下表，其中词性标签 24 个（小写字母），专名类别标签 4 个（大写字母）。

# #并行分词  - Five Step
# 
# 原理：将目标文本按行分隔后，把各行文本分配到多个 Python 进程并行分词，然后归并结果，从而获得分词速度的可观提升
# 
# 基于 python 自带的 multiprocessing 模块，目前暂不支持 Windows
# 
# 用法：
# 
# jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数
# jieba.disable_parallel() # 关闭并行分词模式
# 例子：https://github.com/fxsjy/jieba/blob/master/test/parallel/test_file.py
# 
# 实验结果：在 4 核 3.4GHz Linux 机器上，对金庸全集进行精确分词，获得了 1MB/s 的速度，是单进程版的 3.3 倍。
# 
# 注意：并行分词仅支持默认分词器 jieba.dt 和 jieba.posseg.dt。

# #Tokenize：返回词语在原文的起止位置  - Sixth Step 

# In[ ]:


result = jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))


# #搜索模式

# In[ ]:


result = jieba.tokenize(u'永和服装饰品有限公司', mode='search')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))


# #ChineseAnalyzer for Whoosh 搜索引擎  - Seventh Step 
# 
# 引用： from jieba.analyse import ChineseAnalyzer
# 
# 用法示例：https://github.com/fxsjy/jieba/blob/master/test/test_whoosh.py

# #命令行分词  - Eigth Step
# 
# 使用示例：python -m jieba news.txt > cut_result.txt
# 
# 命令行选项（翻译）：

# In[ ]:


import jieba
jieba.initialize()  # 手动初始化（可选）


# In[ ]:


jieba.add_word('台中') 
#或者 
jieba.suggest_freq('台中', True)


# In[ ]:


#解决方法：强制调低词频

jieba.suggest_freq(('今天', '天气'), True)


# In[ ]:


#或者直接删除该词 
jieba.del_word('今天天气')


# In[ ]:


jieba.cut('丰田太省了', HMM=False)


# In[ ]:


jieba.cut('我们中出了一个叛徒', HMM=False)


# #Acknowledgement:
# 
# Sun Junyi - fxsjy/jieba
# 
# https://github.com/fxsjy/jieba

# ![](https://c.tenor.com/xvo8-YQ78P0AAAAC/porky-pig.gif)tenor.com
