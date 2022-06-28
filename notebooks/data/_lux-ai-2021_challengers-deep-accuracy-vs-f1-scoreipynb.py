#!/usr/bin/env python
# coding: utf-8

# # Precision, Recall, Accuracy, F1 score

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#TP = true positives
#TN = true negatives
#FP = false positives
#FN = false negatives
def recall_func(TP,FN):
    return TP/(TP + FN)
def precision_func(TP, FP):
    return TP/(TP + FP)
def accuracy_func(TP, TN, FP, FN): 
    return (TP + TN)/(TP + FP + TN + FN)
def f1_score_func(precision, recall):
        return 2 * (precision * recall /(precision + recall))
def f1_score_other(TP, FP, FN):
        return 2 * ((TP/(TP + FP)) * (TP/(TP + FN)) /((TP/(TP + FP)) + (TP/(TP + FN))))


# In[ ]:


# predictions
true_p = 5
true_n = 10
false_p = 3
false_n = 2

# precision
prec = precision_func(true_p, false_p)
# recall
rec = recall_func(true_p,false_n)

# main metrics
accuracy = accuracy_func(true_p, true_n, false_p, false_n)
f1_score = f1_score_func(prec, rec)
f1 = f1_score_other(true_p, false_p, false_n)

print('precision: ' + str(prec))
print('recall: ' + str(rec))
print('accuracy: ' + str(round(accuracy,4)))
print('f1_score: ' + str(round(f1_score,4)),',', str(round(f1,4)))


# In[ ]:


case = ['positive', 'negative']
true = [5, 10]
false = [3,2]
df = pd.DataFrame(list(zip(case, true, false)), columns = ['Case', "True", 'False'])


# In[ ]:


# plot a heatmap
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(df[['True', 'False']], annot=True, cmap = 'Purples', annot_kws={"size": 15})

plt.title('\nHeatmap for Number of predictions\n', fontsize = 25) # title with fontsize 20
plt.xlabel('True/False', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Positive/Negative', fontsize = 15) # y-axis label with fontsize 15

plt.show()


# In[ ]:


precision_list = [*range(0,110,10)]
recall_list = [*range(0,110,10)]


# In[ ]:


f1_list = []
for p in precision_list:
    for r in recall_list:
        if p == 0 or r == 0:
            f1_list.append(0)
        else:
            f1_list.append(round(f1_score_func(p,r),4))


# In[ ]:


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]
f1_array = split_list(f1_list, 11)


# In[ ]:


# Create a dataset
df = pd.DataFrame(f1_array, precision_list, recall_list)

# plot a heatmap
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(df, annot=True, cmap = 'Purples', annot_kws={"size": 15}, fmt = '.3f')

plt.title('\nThe change in F1 score due to Recall and Precision\n', fontsize = 25) # title with fontsize 20
plt.xlabel('\nRECALL', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('PRECISION\n', fontsize = 15) # y-axis label with fontsize 15

plt.show()


# In[ ]:


cm = cmap=sns.diverging_palette(-80, 250, as_cmap=True)
df.style.background_gradient(cmap = cm)


# In[ ]:


def magnify():
    return [dict(selector="th",
                 props=[("font-size", "4pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]


# In[ ]:


cmap = sns.color_palette("magma", as_cmap=True)

df.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '6pt'})\
    .set_caption("The F1 score with different values of Recall and Precision")\
    .format(precision=2)\
    .set_table_styles(magnify()).set_table_styles([
                            {"selector":"thead",
                             "props": [("background-color", "indigo"), ("color", "khaki"),
                                       ("font-size", "0.7rem"), ("font-style", "italic")]},
                            {"selector":"th.row_heading",
                             "props": [("background-color", "khaki"), ("color", "indigo"),
                                       ("font-size", "0.7rem"), ("font-style", "italic")]}  
                              ])


# Inspired by: [https://mlu-explain.github.io/precision-recall/](http://)
# 
# References: 
# 
# - Seaborn https://seaborn.pydata.org/generated/seaborn.heatmap.html
# 
# - Heatmaps https://www.python-graph-gallery.com/heatmap/
# 
# - More about precision recall, and f1 score 
# https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
# 
# - Color palettes for talble formattings 
# https://seaborn.pydata.org/tutorial/color_palettes.html
# 
# - Table styling in pandas 
# https://pandas.pydata.org/docs/user_guide/style.html
# 
# - List of colors in pandas
# https://datascientyst.com/full-list-named-colors-pandas-python-matplotlib/
# 
# - How to format headers 
# https://coderzcolumn.com/tutorials/python/simple-guide-to-style-display-of-pandas-dataframes
