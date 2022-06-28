#!/usr/bin/env python
# coding: utf-8

# # Úvod
# V tomto notebooku jsme se zaměřili na počty úmrtí způsobeny Covidem-19. Data jsme vybrali vzhledem k atuální situaci, jelikož se nám zdála stále relevantní. Díky vysokému zájmu o vybraná data, jsou dobře zpracována a stále aktualizována. Cílem práce je zjistit závislosti úmrtí na věku, pohlaví, či kraji ČR.
# Použitá data v tomto notebooku jsou aktualizována ke dni 23.05.2022 a citována v podrobnostech datasetu.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython.display import display, Markdown


# In[ ]:


pd.plotting.register_matplotlib_converters()
pd.set_option('max_rows', 5)

smrt = pd.read_csv("../input/jardaawhifazhgbh/umrti.csv", index_col=None)


# Funkce "head()" zobrazí hlavičku tabulky - v našem případě prvních 5 řádků (díky funkce "set_option" v předchozí buňce).

# In[ ]:


smrt.head()


# Zde je použita funkce `describe()`, která nám zobrazí určité důležité hodnoty.
# Vidíme zde medián například `mean`, celkový počet úmrtí `count`, nejnižší věk zemřelé osoby `min`, nebo nejvyšší věk zemřelé osoby `max`. 

# In[ ]:


pd.set_option('max_rows', 10)
display(Markdown(smrt.vek.describe().round(2).to_markdown()))


# Zde je 10 nejstarších zemřelých osob podle věku. Jména jsou nahrazena `id`, kvůli GDPR. 

# In[ ]:


display(Markdown(smrt['vek'].nlargest(n=10).to_markdown()))


# Funkce `groupby()` zařídí seskupení všech dat podle hodnoty z konkrétního sloupce (v tomto případě podle pohlaví). 

# In[ ]:


smrt.groupby("pohlavi").pohlavi.count()


# Zde je vytvořen graf z dat výstupu minulé funkce. Na ose X jsou indexy a na ose Y jejich počet.
# Můžeme zde vidět, že kvůli onemocnění Covid-19 zemřelo více mužů než žen, a to zhruba o 5500.

# In[ ]:


ax = sns.barplot(x=smrt.groupby("pohlavi").pohlavi.count().index, y=smrt.groupby("pohlavi").pohlavi.count())
ax.set(title='Porovnání množství úmrtí každého pohlaví')
ax.set(xlabel='Pohlaví', ylabel='Množství')
plt.show()


# Opět je použita funkce `groupby()`, nyní podle věku. Nezobrazují se zde všechny věkové hodnoty, jelikož by to zabíralo moc místa. :) 

# In[ ]:


smrt.groupby("vek").vek.count()


# Z grafu vytvořeného podle dat minulé funkce lze vyčíst, že smrt postihla nejčastěji lidi ve věku 65 a více.

# In[ ]:


plt.figure(figsize=(16,8))
ax = sns.lineplot(x=smrt.groupby("vek").vek.count().index, y=smrt.groupby("vek").vek.count())
ax.set(title='Závislost věku na počtu úmrtí')
ax.set(xlabel='Věk', ylabel='Množství')
plt.show()


# Zde máme tabulku popisující počet úmrtí v jednotlivých krajích.
# Kraje jsou označeny ve veřejném datasetu kódem, jehož legenda je [zde](https://cs.wikipedia.org/wiki/CZ-NUTS).

# In[ ]:


pd.set_option('max_rows', 14)
smrt.groupby("kraj_nuts_kod").vek.count()


# Tak jak jsme si vytvořili tabulku z dat určitých krajů, tak jsme sestavili taktéž graf. Z grafu jde vyčíst, že nejvíce lidí zemřelo v Moravskoslezském či Jihomoravském kraji.
# Legenda ke kódům krajů znova [zde](https://cs.wikipedia.org/wiki/CZ-NUTS).

# In[ ]:


plt.figure(figsize=(16,8))
ax = sns.barplot(x=smrt.groupby("kraj_nuts_kod").vek.count().index, y=smrt.groupby("kraj_nuts_kod").vek.count())
ax.set(title='Počet úmrtí v závislosti na kraji')
ax.set(xlabel='Kraje', ylabel='Množství')
plt.show()


# # Závěr
# Závěrem našeho notebooku je shrnutí všech cílů, které jsme si vytyčili v úvodu. Jedná se o shrnutí absolutních čísel jako medián a celkový počet všech úmrtí, či nejstarší zemřelou osobu. Sestavili jsme tabulku nejstarších zemřelých osob. A hlavně jsme graficky zobrazili porovnání množství úmrtí každého pohlaví, závislost věku na počtu úmrtí, nebo počet úmrtí v závislosti na kraji. Z těchto dat jsme vyčetli, že na COVID-19 zemřel větší množství mužů než žen, smrtelně postihl COVID-19 spíš lidi starší 65 let a nejvíce lidí zemřelo v kraji Moravskoslezském. Dataset je nastaven tak, aby se každý týden aktualizoval a my jsme tak schopni zobrazit stále aktuální data.
