#!/usr/bin/env python
# coding: utf-8

# ![bigstock-181459699-min-1.jpg](attachment:4b3f677a-9fdb-40d6-b018-13996f835f96.jpg)
# image from https://selfhacked.com/blog/symptoms-low-high-hemoglobin-diseases-increase-decrease/
# 
# I read the article 'Effect of Cigarette Smoking on Haematological Parameters in Healthy Population' which shows that continuous cigarette smoking has severe adverse effects on haematological parameters (e.g., hemoglobin, white blood cells count, mean corpuscular volume, mean corpuscular hemoglobin concentration, red blood cells count, hematocrit) and these alterations might be associated with a greater risk for developing atherosclerosis, polycythemia vera, chronic obstructive pulmonary disease and/or cardiovascular diseases.
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5511531/
# 
# So in this notebook, I tried to find the correlation between 'smoking' and 'hemoglobin' or/and any other features. 
# 
# And as next step, I also tried 'smoker finding models' which can tell us who is smoking from features.

# # Importing

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('/kaggle/input/body-signal-of-smoking/smoking.csv')


# # Data Outline

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


num_list = list(df.columns)

fig = plt.figure(figsize=(10,30))

for i in range(len(num_list)):
    plt.subplot(15,2,i+1)
    plt.title(num_list[i])
    plt.hist(df[num_list[i]],color='blue',alpha=0.5)

plt.tight_layout()


# # Feature Engineering - handling outliers

# * The features which seem to have outliers

# In[ ]:


num_list3=['eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)',
       'triglyceride', 'HDL', 'LDL', 'Urine protein',
       'serum creatinine', 'AST', 'ALT', 'Gtp']


# * Find the value of 99.5%

# In[ ]:


for i in range(len(num_list3)):
    print(df[num_list3[i]].quantile(0.995))


# * Update the features within 99.5%

# In[ ]:


df=df[df['eyesight(left)'] < 2]
df=df[df['eyesight(right)'] < 2]
df=df[df['hearing(left)'] < 2]
df=df[df['hearing(right)'] < 2]
df=df[df['triglyceride'] <= 381]
df=df[df['HDL'] <=106]
df=df[df['LDL'] <=216]
df=df[df['AST'] <=103.54499999999825]
df=df[df['ALT'] <=137.0]
df=df[df['Gtp'] <= 328]


# In[ ]:


df.describe().T


# # Visualization

# In[ ]:


num_list4=['age','height(cm)', 'weight(kg)', 'waist(cm)',
       'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)',
       'systolic', 'relaxation', 'fasting blood sugar', 'Cholesterol',
       'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
       'serum creatinine', 'AST', 'ALT', 'Gtp']
fig = plt.figure(figsize=(10,40))

for i in range(len(num_list4)):
    plt.subplot(11,2,i+1)
    plt.title(num_list4[i])
    plt.violinplot(df[num_list4[i]])

plt.tight_layout()


# In[ ]:


sns.pairplot(df,hue='smoking',vars=['systolic', 'relaxation', 'fasting blood sugar', 'Cholesterol',
       'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein',
       'serum creatinine', 'AST', 'ALT', 'Gtp'])


# In[ ]:


df=pd.get_dummies(df)


# In[ ]:


plt.figure(figsize = (15,8))
sns.heatmap(df.corr(),annot=True, cbar=False, cmap='Blues', fmt='.1f')


# Actulally, correlation between 'smoking' and 'hemoglobin' is higher than any other medical features other than gender or height.

# In[ ]:


df1=df.groupby('smoking').mean().T
df1['gap'] = df1.apply(lambda x: (x[1] - x[0])/x[1]*100, axis = 1)
df1.sort_values(by="gap",ascending=False)


# By comapring the averege between 'no smoking' and 'smoking', we can find gap in 
# * Gender_M 53.3%
# * Gtp 41.4%
# * dental caries 33.5%
# * triglyceride 24.5%
# * ALT 19.7%
# * tartar_Y 16.4%
# * weight(kg) 11.4%
# * serum creatinine 10.4%
# * hemoglobin 8.5%

# # Comapring Models

# In[ ]:


pip install catboost


# In[ ]:


x=df.drop(['ID','smoking'],axis=1)
y=df['smoking']


# In[ ]:


from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[ ]:


x_norm = (x - np.min(x)) / (np.max(x)).values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x_norm,y,test_size=0.3,random_state=42)
method_names = []
method_scores = []


# * LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
print("Logistic Regression Classification Test Accuracy {}".format(log_reg.score(X_test,y_test)))
method_names.append("Logistic Reg.")
method_scores.append(log_reg.score(X_test,y_test))

y_pred = log_reg.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# * KNN (K-Nearest Neighbour) CLASSIFICATION

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
print("Score for Number of Neighbors = 8: {}".format(knn.score(X_test,y_test)))
method_names.append("KNN")
method_scores.append(knn.score(X_test,y_test))

y_pred = knn.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# * SUPPORT VECTOR MACHINE (SVM)

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(X_train,y_train)
print("SVM Classification Score is: {}".format(svm.score(X_test,y_test)))
method_names.append("SVM")
method_scores.append(svm.score(X_test,y_test))

y_pred = svm.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# * NAIVE BAYES

# In[ ]:


from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_test,y_test)
print("Naive Bayes Classification Score: {}".format(naive_bayes.score(X_test,y_test)))
method_names.append("Naive Bayes")
method_scores.append(naive_bayes.score(X_test,y_test))

#Confusion Matrix
y_pred = naive_bayes.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# * DECISION TREE

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train,y_train)
print("Decision Tree Classification Score: ",dec_tree.score(X_test,y_test))
method_names.append("Decision Tree")
method_scores.append(dec_tree.score(X_test,y_test))

y_pred = dec_tree.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# * RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(X_train,y_train)
print("Random Forest Classification Score: ",rand_forest.score(X_test,y_test))
method_names.append("Random Forest")
method_scores.append(rand_forest.score(X_test,y_test))

y_pred = rand_forest.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# * CATBOOST

# In[ ]:


import catboost as cb
clf = cb.CatBoostClassifier()
clf.fit(X_train, y_train)
print("CatBoost Score: ",clf.score(X_test,y_test))
method_names.append("CatBoost")
method_scores.append(clf.score(X_test,y_test))

y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
plt.ylim([0.5,0.90])
plt.bar(method_names,method_scores,width=0.3)
plt.xlabel('Method Name')
plt.ylabel('Method Score')


# By comparing 7 models, RandomForest seems to be the best with accuracy 82.2%.

# Thank you !
