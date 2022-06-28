#!/usr/bin/env python
# coding: utf-8

# Tato práce zobrazuje data výherců v F1 a následně je zpracovává. Cílem je zjistit, jaký vliv mají určité okolnosti na následné umístění v závodech.

# In[ ]:


import pandas as pd
import numpy as np
drivers= pd.read_csv("../input/f1winners/F1 Winners Dataset.csv", index_col=0)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


drivers


# In[ ]:


drivers.iloc[0:10]


# Tabulka top 10 závodníků

# In[ ]:


round(drivers.Wins.mean(),2)


# Průměrný počet výher na závodníka.

# In[ ]:


drivers.Podiums.median()


# Medián počtu umístění na pódiu

# In[ ]:


max(drivers.Wins)


# Největší počet výher

# In[ ]:


drivers.Country.value_counts()


# Tabulka zobrazující počet závodníků z každé země

# In[ ]:


drivers.groupby(['Country']).Wins.agg([max,min])


# Tabulka zobrazující maximální a minimální počet výher závodníků vybrané země

# In[ ]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=drivers["GPs Entered"], y=drivers["Wins"])


# Graf závislosti počtu účastí na Grand Prix na počtu výher

# In[ ]:


plt.figure(figsize=(25,10))
sns.swarmplot(x=drivers["Championships"], y=drivers["Wins"])


# Graf ukazující souvislost mezi počtem mistrovstí světa jezdců a počtem výher

# In[ ]:


plt.figure(figsize=(12,6))
sns.lmplot(x="Poles", y="Wins", data=drivers)


# Graf ukazující vazbu mezi počtem Pole position (start z 1. místa) a počtem výher

# In[ ]:


plt.figure(figsize=(12,6))
sns.kdeplot(x=drivers["GPs Entered"],y=drivers["Podiums"], shade=True).set(xlim=1,ylim=1)


# Graf ukazující závislost účastí na Grand Prix a počtem umístění na stupních vítězů

# Závěrem této práce je zjištění, že:
# * Největší počet výherců je z Velké Británie
# * Závodník s největším počtem výher je Lewis Hamilton (počet výher 103)
# * S rostoucím počtem účastí na Grand Prix roste počet výher
# * Závodníci s největším počtem šampionátů mají nejvíce výher
# * Závodníci, kteří začínají na lepší startovní pozici mají větší šanci na výhru
# 
# V práci se taky můžete dovědět průměrný počet výher na závodníka nebo medián počtu umístění na pódiu 
# 
