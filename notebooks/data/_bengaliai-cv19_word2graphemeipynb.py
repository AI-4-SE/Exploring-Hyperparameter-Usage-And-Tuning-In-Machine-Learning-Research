#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


path_lexicon = '../input/indicword/IndicWord/IndicWord/Bangla/Bangla_lexicon.txt'


# In[ ]:


with open(path_lexicon,'r',encoding='utf-16') as f:
    lexicon = [word.strip() for word in f.readlines()]
# lexicon


# In[ ]:


path_class_map = '../input/bengaliai-cv19/class_map.csv'
df_map = pd.read_csv(path_class_map)
df_map.head()


# In[ ]:


df_root = df_map.groupby('component_type').get_group('grapheme_root')
df_root.index = df_root['label']
df_root = df_root.drop(columns = ['label','component_type'])
df_root.head()


# In[ ]:


df_vd = df_map.groupby('component_type').get_group('vowel_diacritic')
df_vd.index = df_vd['label']
df_vd = df_vd.drop(columns = ['label','component_type'])
df_vd


# In[ ]:


df_cd = df_map.groupby('component_type').get_group('consonant_diacritic')
df_cd.index = df_cd['label']
df_cd = df_cd.drop(columns = ['label','component_type'])
df_cd


# In[ ]:


word = 'র্য'
print('{} = {}'.format(word, [u for u in word]))


# In[ ]:


for wd in lexicon[:50]:
    print('word: {}, unicode elements: {}'.format(wd,[uc for uc in wd]))


# In[ ]:


def word2grapheme(word):
    
    graphemes = []
    grapheme = ''
    i = 0
    while i < len(word):    
        grapheme+=(word[i])
#         print(word[i], grapheme, graphemes)
        # deciding if the grapheme has ended
        if word[i] in ['\u200d','্']:
            # these denote the grapheme is contnuing
            pass
        elif  word[i] == 'ঁ': # 'ঁ' always stays at the end
            graphemes.append(grapheme)
            grapheme = ''
        elif word[i] in list(df_root.values)+ ['়']: 
            # root is generally followed by the diacritics
            # if there are trailing diacritics, don't end it
            if i+1 ==len(word):
                graphemes.append(grapheme)
            elif word[i+1] not in ['্', '\u200d', 'ঁ', '়'] + list(df_vd.values):
                # if there are no trailing diacritics end it
                graphemes.append(grapheme)
                grapheme = ''

        elif word[i] in df_vd.values: 
            # if the current character is a vowel diacritic
            # end it if there's no trailing 'ঁ' + diacritics
            # Note: vowel diacritics are always placed after consonants 
            if i+1 ==len(word):
                graphemes.append(grapheme)
            elif word[i+1] not in ['ঁ'] + list(df_vd.values):
                graphemes.append(grapheme)
                grapheme = ''                

        i = i+1
        # Note: df_cd's are constructed by df_root + '্'
        # so, df_cd is not used in the code

    return graphemes
    


# In[ ]:


word_list = ['আর্দ্র','ওরা','হিজড়াদের','শ্য়পূ','আবহাওয়াবিদ্যা','প্রকাশ্যে',
             'প্রিথ্রীব্রি', 'য়ন্তে', 'য়র্সে','ধ্য়য়নে', 'খ্য়যো', 'মায়াবি', '়য়বা',
             'ন্দর্যে', 'সৌন্দর্য','উৎকৃষ্টতম্', 'হতাে','ফেক্সােফেনাডিন', 'জুতশীঅশােক',
            'টাের্মিনাসগুলির', 'র্যােককুন']
# word_list = [ ]
for wd in word_list:
    print('word: {}, unicode elements: {}, graphemes: {}'.format(
         wd,[uc for uc in wd], word2grapheme(wd)))


# In[ ]:


for wd in lexicon:
    if 'ঁ' in [uc for uc in wd]:
        print('word: {}, unicode elements: {}, graphemes: {}'.format(wd,[uc for uc in wd], word2grapheme(wd)))


# In[ ]:


df = pd.DataFrame(
    data = {'lexicon': lexicon, 'graphemes': ['--'.join(word2grapheme(wd.replace('-', ''))) for wd in lexicon]},
    columns = ['lexicon','graphemes']
)


# In[ ]:


df.head(50)


# In[ ]:


df.to_csv('indicword_bengali_lexicon_grapheme.csv')


# In[ ]:


word = 'দুুর্জন'


# In[ ]:


[u for u in word]


# In[ ]:


word2grapheme(word)

