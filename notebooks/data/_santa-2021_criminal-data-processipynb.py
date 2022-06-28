#!/usr/bin/env python
# coding: utf-8

# ## 歷年犯罪資料統計數據
# ### [dataset](https://data.gov.tw/dataset/13166)

# In[ ]:


get_ipython().system('pip install pyexcel_ods3')
'''
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-2-56aa90807b82> in <module>
      1 import pandas as pd
----> 2 from pyexcel_ods3 import get_data
      3 import re
      4 from pathlib import Path
      5 import arrow

ModuleNotFoundError: No module named 'pyexcel_ods3
'''


# In[ ]:


import pandas as pd
from pyexcel_ods3 import get_data
import re
from pathlib import Path
import arrow
from ipywidgets import IntProgress
from IPython.display import display


# In[ ]:


data = get_data(r"../input/taiwan-crime-statistics/107年10月15日至107年10月21日資料.ods")
data
print(type(data))
display(data)


# In[ ]:


for k,v in data.items():
    print(type(k))
    display(k)
    print(type(v))
    display(v)


# In[ ]:


for k,v in data.items():
    i = 0
    for vi in v:
        print(f"v[{i}] type is {type(vi)}")
        display(vi)
        i += 1


# In[ ]:


for k,v in data.items():
    column = v[2] # 案類別
    for i in range(0, len(column)):
        col=  column[i]  
        print(f"column[{i}]  '{col}'\t {type(col)}\t  length {len(col)}")    


# In[ ]:


df_column = []
df_column.append("日期")
for k,v in data.items():
    column = v[2] #案類別
    for i in range(1, len(column)):
        col=  column[i]  
        print(f"column[{i}]  {col}  length {len(col)}")
        if (len(col) !=0) :
            df_column.append(col+"發生數")
            df_column.append(col+"破獲數")
            print(col+"發生數")
            print(col+"破獲數")  
df_column


# In[ ]:


for k,v in data.items():
    cases = v[3]    # 發生數
    solvedes = v[4] # 破獲數 
    for i in range(0,len(cases)):
        case =  cases[i]  
        print(f"cases[{i}]  {case}")  
    for i in range(0,len(solvedes)):
        solved = solvedes[i]
        print(f"solveds[{i}]  {solved}")      


# In[ ]:


df_cases = []
for k,v in data.items():
    cases = v[3]    # 發生數
    solvedes = v[4] # 破獲數
    print(f"發生數 :")
    index = 0
    for i in range(1,len(cases)):
        case = cases[i]
        print(f"'{case}' {type(case)}, ", end ='')
        if (isinstance(case, int)):
            df_cases.insert(index*2, case) # case at index 0, 2, 4, 6 ... index*2
            index += 1        
    print(f"\n 破獲數 :")
    inex = 0
    for i in range(1,len(solvedes)):
        solved = solvedes[i]
        print(f"'{solved}' {type(case)}, ", end = '')
        if(isinstance(solved, int)):
            df_cases.insert(index*2+1, solved) # # solved at index 1, 3, 5, 7 ... index*2+1
            index += 1
    print()
print(f"{cases} \n{solvedes} \n")    
df_cases


# ## Take 年月日
# ### 正規表示式 
# 
# [簡易 Regular Expression 入門指南](https://blog.techbridge.cc/2020/05/14/introduction-to-regular-expression/)
# 
# [Regular Expression (regex)，要成為GA專家一定要懂的正規表示式](https://transbiz.com.tw/regex-regular-expression-ga-%E6%AD%A3%E8%A6%8F%E8%A1%A8%E7%A4%BA%E5%BC%8F/)
# 
# [regex101](https://regex101.com/)
# 
# \*【星號】比對前一個字串零次或是多次。 
# 
# \+ 【加號】至少要與前一個字比對一次或以上。
# 
# ( )【括號】 把你想要找的相關字詞放在括號裡面，
# 
# [ ] 【中括號】
# 任意比對字串中裡面的每個項目。比如你設定[DEFG]
# 

# In[ ]:


for k,v in data.items():
    dates = v[1][0] # 中華民國107年10月15日至107年10月21日 --> 中華民國107年10月15日 --> 10/15/2018
    match = re.search("中華民國(\s*\d+\s*)年(\s*\d+\s*)月(\s*\d+\s*)日", dates)
    mg = match.groups()
    date_str = mg[1]+"/"+mg[2]+"/"+ str(int(mg[0])+1911)
    print(date_str)


# In[ ]:


df_cases = []
for k,v in data.items():
    ## Add case and solveds 
    cases = v[3]    # 發生數
    solvedes = v[4] # 破獲數
    #print(f"發生數 :")
    index = 0
    for i in range(1,len(cases)):
        case = cases[i]
        #print(f"'{case}' {type(case)}, ", end ='')
        if (isinstance(case, int)):
            df_cases.insert(index*2, case) # case at index 0, 2, 4, 6 ... index*2
            index += 1        
    #print(f"\n 破獲數 :")
    index = 0
    for i in range(1,len(solvedes)):
        solved = solvedes[i]
        #print(f"'{solved}' {type(case)}, ", end = '')
        if(isinstance(solved, int)):
            df_cases.insert(index*2+1, solved) # # solved at index 1, 3, 5, 7 ... index*2+1
            index += 1
    #print()
    ## Add date_time at index 0
    dates = v[1][0] # 中華民國107年10月15日至107年10月21日 --> 中華民國107年10月15日 --> 10/15/2018
    match = re.search("中華民國(\s*\d+\s*)年(\s*\d+\s*)月(\s*\d+\s*)日", dates)
    mg = match.groups()
    date_str = mg[1]+"/"+mg[2]+"/"+ str(int(mg[0])+1911)
    #print(date_str)
    df_cases.insert(0, date_str)   
#print(f"{cases} \n{solvedes} \n")    
df_cases


# In[ ]:


# make dataframe from one file
df_data=[df_cases]
df1 = pd.DataFrame(df_data, columns = df_column)
df1


# In[ ]:


# list the all files of the path
allfiles = list(Path('../input/taiwan-crime-statistics/').glob("*.ods"))
allfiles


# In[ ]:


def make_dfcases(file_name):
    data = get_data(file_name)
    df_cases = []
    for k,v in data.items():
        ## Add case and solveds
        cases = v[3]    # 發生數
        solvedes = v[4] # 破獲數
        #print(f"發生數 :")
        index = 0
        for i in range(1,len(cases)):
            case = cases[i]
            #print(f"'{case}' {type(case)}, ", end ='')
            if (isinstance(case, int)):
                df_cases.insert(index*2, case) # case at index 0, 2, 4, 6 ... index*2
                index += 1        
        #print(f"\n 破獲數 :")
        index = 0
        for i in range(1,len(solvedes)):
            solved = solvedes[i]
            #print(f"'{solved}' {type(case)}, ", end = '')
            if(isinstance(solved, int)):
                df_cases.insert(index*2+1, solved) # # solved at index 1, 3, 5, 7 ... index*2+1
                index += 1
        #print()
        ## Add date_time at index 0
        dates = v[1][0] # 中華民國107年10月15日至107年10月21日 --> 中華民國107年10月15日 --> 10/15/2018
        match = re.search("中華民國(\s*\d+\s*)年(\s*\d+\s*)月(\s*\d+\s*)日", dates)
        mg = match.groups()
        date_str = mg[1]+"/"+mg[2]+"/"+ str(int(mg[0])+1911)
        #print(date_str)
        df_cases.insert(0, date_str)   
    #print(f"{cases} \n{solvedes} \n")
    return df_cases


# In[ ]:


case_list = make_dfcases(r"../input/taiwan-crime-statistics/107年10月15日至107年10月21日資料.ods")
print(case_list)


# In[ ]:


# transform all files data to dataframe 
allfiles = list(Path('../input/taiwan-crime-statistics/').glob("*.ods"))
df_data = []
# Initialize a progess bar
progress = IntProgress()
progress.max = len(allfiles)
progress.description = '(Init)'
display(progress)

for file in allfiles:
    #print(str(file))
    case = make_dfcases(str(file))
    if (len(case) == len(df_column)):
        df_data.append(case)
    progress.value += 1
    progress.description = "Processing"
    
progress.description = 'Done'    
df_data


# In[ ]:


df = pd.DataFrame(df_data, columns = df_column)
df


# In[ ]:


df.info()


# In[ ]:


df['日期'] = pd.to_datetime(df['日期'])
df.info()


# In[ ]:


df.sort_values(by=['日期'], inplace=True)
df


# In[ ]:


res = df[df['日期'].between('2020-1-1', '2020-12-31')].sum()
print(f"{type(res)} \n\nThe summary of 2020")
display(res)


# In[ ]:


start = arrow.get('2021-1-1')
end = arrow.get('2021-12-31')
end = end.shift(days=-1)

sdate = str(start.date())
edate = str(end.date())

print(f"{type(res)} \n\nThe summary from {sdate} to {edate}")
res = df[df['日期'].between(sdate, edate)]
res


# In[ ]:


start = arrow.get('2018-1-1')
end = start.shift(months=11)
print(f"start_date : {start.date()}  end_date{str(end.date())}")
sum_data = []
for r in arrow.Arrow.interval('month', start, end, 3):
    start_date = str(r[0].date())
    end_date  = str(r[1].date()) 
    print (f"{start_date} ,{end_date} ")    
    sum1 = df[df['日期'].between(start_date, end_date)].sum()
    sum_data.append(start_date)
    sum_data.append(sum1)
print("====================================================")    
for i in sum_data:
    print(i)
    print("===================================================")       


# In[ ]:


df['Month'] = df['日期'].dt.month 
seasoni = df[df['Month'].between(1,3)].sum()
seasoni = seasoni / 4

seasonii = df[df['Month'].between(4,6)].sum()
seasonii = seasonii / 4

seasoniii = df[df['Month'].between(7,9)].sum()
seasoniii = seasoniii / 4

seasoniv = df[df['Month'].between(10,12)].sum()
seasoniv = seasoniv / 4

print('2018-2021第一季')
print(seasoni,end = "\n")

print('2018-2021第二季')
print(seasonii,end = "\n")

print('2018-2021第三季')
print(seasoniii,end = "\n")

print('2018-2021第四季')
print(seasoniv,end = "\n")

