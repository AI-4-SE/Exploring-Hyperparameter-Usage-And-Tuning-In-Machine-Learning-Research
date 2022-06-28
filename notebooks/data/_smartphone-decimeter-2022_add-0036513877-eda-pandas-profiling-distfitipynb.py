#!/usr/bin/env python
# coding: utf-8

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


# Load data
df = pd.read_csv("../input/dapprojekt22/train.csv")
df_random = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
df.head()


# # Pandas Profiling: brz pregled podataka
# 
# Pozivom jedne metode ProfileReport možemo dobiti lijepo formatirani pregled i opis podataka.

# In[ ]:


from pandas_profiling import ProfileReport


# **Primjer na malom skupu**

# In[ ]:


profile = ProfileReport(df_random, title='Pandas Profiling Report', html={'style': {'full_width': True}})
profile


# **Pandas Profiling na NBA skupu**

# In[ ]:


profile = ProfileReport(df, title='Pandas Profiling Report', html={'style': {'full_width': True}}, minimal=True)
profile


# **Kratko čišćenje prije daljega**
# 
# Micanje stupaca u kojima su prisutne nedostajuće vrijednosti te micanje stupaca sa konstantnim vrijednostima.

# In[ ]:


# izbrisi stupce koje imaju nedostajuće vrijednosti
df.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)

# izbrisi stupce koje imaju konstantne vrijednosti
unique_values = df.nunique()
constant_columns = []
for k,v in unique_values.items():
    if v==1:
        constant_columns.append(k)

df.drop(labels=constant_columns, axis=1, inplace=True, errors='raise')

df.head()


# # Distribucije značajki
# 
# Od velike je koristi promatrati kako su pojedine značajke distribuirane. Biblioteka distfit je korisna za takve stvari.

# In[ ]:


get_ipython().system('pip install distfit')
from distfit import distfit


# **Primjer na jednostavnim podacima generiranima iz jedinične normalne razdiobe**

# In[ ]:


dist = distfit()
X = np.random.randn(10000)
dist.fit_transform(X, verbose=1)

print(dist.summary)
dist.plot_summary()


# **Distribucije značajki iz NBA skupa**

# In[ ]:


distributions = {}


for col in df.columns:
    x = df[col]
    dist = distfit()
    try:
        # nece proci ako nisu numericke znacajke
        dist.fit_transform(X=x, verbose=1)
        best_distribution = str(dist.summary.sort_values('score')['distr'][0])
    except:
        best_distribution = 'INVALID'
    distributions[col] = best_distribution


# In[ ]:


pd.DataFrame.from_dict(distributions, orient='index', columns=['Distribucija'])


# **Broj značajki po distribucijama**
# 
# Za svaku distribuciju izbrojimo koliko značajki se ravna prema toj distribuciji, kako je procijenio distfit.

# In[ ]:


distributions_features = {}
count_distributions = {}

for f,d in distributions.items():
    if d in distributions_features:
        distributions_features[d].append(f)
    else:
        distributions_features[d] = [f]
    
    if d in count_distributions:
        count_distributions[d] = count_distributions[d]+1
    else:
        count_distributions[d] = 1

pd.DataFrame.from_dict(count_distributions, orient='index', columns=['Broj pojavljivanja'])


# In[ ]:


distributions_features['norm']


# Zanimljive su značajke koje imaju uniformnu distribuciju ili je došlo do pogreške prilikom procjene distribucije.

# In[ ]:


distributions_features['uniform']


# Do pogreške je došlo jer značajke nisu realni brojevi, nego znakovni nizovi ili nekakvo vremensko trajanje.

# In[ ]:


distributions_features['INVALID']


# In[ ]:


import plotly.express as px


# In[ ]:


def histogram(data, column='', nbins = 30, use_nbins = False):
        
    coldata = data[column]
    
    if use_nbins:
        fig = px.histogram(data, x=column, nbins=nbins)
    else:
        fig = px.histogram(data, x=column)
    return fig


# In[ ]:


fig = histogram(df, column='AST_RATIO_HOME')
fig.show()


# In[ ]:


fig = histogram(df, column=distributions_features['beta'][6])
fig.show()


