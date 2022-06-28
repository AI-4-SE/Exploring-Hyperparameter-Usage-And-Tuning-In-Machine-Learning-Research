#!/usr/bin/env python
# coding: utf-8

# # Is Valentino Rossi the most successful racer in MotoGP?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


pip install pygal


# In[ ]:


pip install squarify


# In[ ]:


import plotly as plot
import pygal as py
import squarify as sq
import matplotlib 
plt.rcParams["figure.figsize"] = (20,15)
matplotlib.rc('xtick', labelsize=7) 
matplotlib.rc('ytick', labelsize=7) 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}

matplotlib.rc('font', **font)


# In[ ]:


df = pd.read_csv("../input/moto-gp-world-championship19492022/riders-info.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.rename(columns={"Riders All Time in All Classes":"Name"}, inplace=True)


# In[ ]:


df1 = df.fillna(0)


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
comment_words = ''
stopwords = set(STOPWORDS)
 
for val in df1.Name:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
                    
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[ ]:


df1.shape


# In[ ]:


df1.describe().transpose()


# In[ ]:


df1.plot.box()


# In[ ]:


df1.head(25).plot.barh(x="Name",y="Victories",color="red",alpha=0.80);plt.xlabel("victories");plt.ylabel("Name of Players");
plt.title("Victories by Players");plt.show()


# In[ ]:


df1[df1["World Championships"]>0].plot(x="Name",y="World Championships",kind="bar",color="green")
plt.xlabel("Player name")
plt.ylabel("No . of world campionships")
plt.title("No. of world title won by players",fontsize=10,pad=5)
plt.show()


# In[ ]:


w_c = df1[df1["World Championships"]>0]

norm = matplotlib.colors.Normalize(vmin=min(w_c["World Championships"]), vmax=max(w_c["World Championships"]))
colors = [matplotlib.cm.Blues(norm(value)) for value in w_c["World Championships"]]

fig = plt.gcf()
ax = fig.add_subplot()

sq.plot(label=w_c.Name,sizes=w_c.Victories, color = colors, alpha=.6,pad = True)
plt.title("Pemenang kejuaraan Motogp dengan ukuran mereka menunjukkan kemenangan",fontsize=23,fontweight="bold")

plt.axis('off')
plt.show()


# In[ ]:


df_mt=df1.drop(columns=["Pole positions from '74 to 2022","Race fastest lap to 2022","World Championships"])

df_mt.head(10).plot(x='Name', kind='bar', stacked=True,
        title='Stacked Bar Graph by dataframe')

plt.xlabel("Name of moto gp players")
plt.ylabel("WINNERS in 1 st,2nd,3rd position")
plt.show()


# In[ ]:


total = sum(df1["Victories"])
data = [sum(df1["Victories"].head(10)),sum(df1["Victories"])-sum(df1["Victories"].head(10))]

sizes = data
labels = ['top 10 player victories', 'Other players victories']
colors = ['blue', 'red']

explode = (0.05, 0.05)

# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,autopct='%1.1f%%', pctdistance=0.85,explode=explode)

# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title('Victories of top 10 players vs other players')

plt.legend(labels, loc="upper left", title="Comparision of top players vs others")

plt.show()


# In[ ]:


corr=df1.corr()
corr.style.highlight_max(color="red",axis=1)


# In[ ]:


sns.heatmap(corr)


# In[ ]:


pip install klib


# In[ ]:


import klib


# In[ ]:


klib.corr_plot(df1, split='pos')


# In[ ]:


klib.dist_plot(df1)
plt.show()


# In[ ]:


klib.cat_plot(df1, top=4, bottom=4)
plt.show()


# In[ ]:


plt.figure(facecolor="olive",edgecolor="green")
sns.set_palette( 'inferno_r')
sns.set_style("darkgrid")
df2=df1.sort_values(by="Victories",ascending=False).head(10)
df3=df1.sort_values(by="2nd places",ascending=False).head(10)
df4=df1.sort_values(by="3rd places",ascending=False).head(10)
df5=df1.sort_values(by="World Championships",ascending=False).head(10)
fig, axes = plt.subplots(4,1)
fig.suptitle('Players top in their positions')


sns.barplot(ax=axes[0], x=df2.Name, y=df2.Victories)
axes[0].set_title("players with highest victories")


sns.barplot(ax=axes[1], x=df3.Name, y=df3["2nd places"])
axes[1].set_title("Players with most no. of 2 nd positions")


sns.barplot(ax=axes[2], x=df4.Name, y=df4["3rd places"])
axes[2].set_title("Players with most no. of 3rd positions")

sns.barplot(ax=axes[3], x=df5.Name, y=df5["World Championships"])
axes[3].set_title("Players with most no. of World Campionships")

plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.pairplot(df1,palette="rainbow",corner=True,plot_kws=dict(marker="+", linewidth=1),diag_kws=dict(fill=False));plt.show()


# In[ ]:


import plotly.express as px
fig = px.box(df1, y="Race fastest lap to 2022", points="all",notched=True)
fig.show()


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Box(x=df1["Victories"],name='1st places'))
fig.add_trace(go.Box(x=df1["2nd places"],name="2nd places"))
fig.add_trace(go.Box(x=df1["3rd places"],name="3rd place"))
fig.add_trace(go.Box(x=df1["World Championships"],name="World Championships"))

fig.update_layout(title_text="Box ploting posisi pembalap dalam data yang diberikan")
fig.update_traces(orientation='h')
fig.show()


# In[ ]:


fig = px.scatter(df1, x="Pole positions from '74 to 2022", y="World Championships", size='Victories',color="Race fastest lap to 2022")
fig.show()


# In[ ]:


sns.lmplot(data=df1, x="Pole positions from '74 to 2022", y="World Championships",markers=["*"],palette="Set1")


# In[ ]:


fig2= px.treemap(data_frame=df1, path=["Name","Victories","2nd places","3rd places"],
                values='Victories',color='World Championships', hover_data=["Race fastest lap to 2022"],color_continuous_scale='RdBu',
                color_continuous_midpoint=np.average(df1["World Championships"], weights=df1['Victories']))
fig2.update_traces(root_color="cyan")
fig2.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig2.show()


# In[ ]:


w_v_m=df1.drop(columns=["2nd places","3rd places","Race fastest lap to 2022","Name"])  
w_v_m.loc[w_v_m['World Championships'] <= 0, 'Won Championship or not?'] = 'False' 
w_v_m.loc[w_v_m['World Championships'] > 0, 'Won Championship or not?'] = 'True' 
w_v_m


# In[ ]:


X=w_v_m.iloc[:,:2]
y=w_v_m["Won Championship or not?"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
  
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
y_pred = gnb.predict(X_test)
  
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

accuracy_score(y_test, y_pred)


# In[ ]:


from sklearn.metrics import confusion_matrix
 
expected = y_test
predicted = y_pred
results = confusion_matrix(expected, predicted)
print(results)


# In[ ]:


y_pred = gnb.predict([[5,10]]) #Contoh Prediksi
y_pred


# * Giacomo Agostini became the most successful racer in MotoGP, Valentino Rossi the second most successful racer in MotoGP.
# 
# * However, Valentino Rossi became the racer with the highest number of wins (Podium).
# 
# * Marc Marquez is the most successful MotoGP racer who is still active.
# 
# * About 40% of the championships are won by these top seeded players.
# 
# * More than 50% of the wins went to the top 10 players.
# 
# * Winning the championship is much more difficult than getting the win, 2nd or 3rd place.
# 
# * Because the number of pole positions increases the chances of winning the world championship and getting a win also increases.
# 
# * The world championship depends on the number of pole positions and wins.
