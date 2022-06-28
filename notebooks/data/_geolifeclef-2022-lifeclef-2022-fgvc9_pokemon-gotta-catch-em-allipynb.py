#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isna().sum()


# Type 2 is filled with null values, filling them with a string of "None" would make more sense as nothing cannot be a type.

# In[ ]:


df = df.fillna('None')


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# ## Lets look at the distribution in the class types of the pokemon from the dataset.

# In[ ]:


px.pie(df, names = "Type 1")


# In[ ]:


px.pie(df, names = "Type 2")


# In[ ]:


px.density_heatmap(df, x="Type 1", y="Type 2", marginal_x="histogram", marginal_y="histogram")


# ## Here we can see some overlap between the two Types that every pokemon has.

# In[ ]:


ff.create_distplot([df.Total],['Total'], bin_size = 10)


# ## The 2 peaks are caused by the evolution of younger pokemons. Not all pokemon have 2 evolutions and some never evolove or are on maximum evolution from the get, this is the reason why there are more pokemon with more than the baseline evolution.

# In[ ]:


fig = px.sunburst(df, path=[px.Constant("Pokemon Gens"), 'Generation', 'Type 1'], color = 'Total',
                    color_continuous_scale='Rainbow',
                    color_continuous_midpoint=df.Total.mean()
                 )
fig.update_layout(
    title_text = 'Distribution of Pokemon types per generation'
)

fig.show()


# In[ ]:


fig = px.bar(df, x="Type 1", color="Type 1", animation_frame="Generation")
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
fig.show()


# ## We can see the number of each time changing based on different generations. Different games are set in different regions and hence have different proportions of pokemon in them.

# In[ ]:


fig = px.scatter(df, x="Type 1", y="Total", color="Type 1",
          animation_frame="Generation", animation_group="Type 2",
          hover_name = 'Name')
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
fig.show()


# ## Can you spot the Legendary pokemons? The outliers are the legendary pokemons! We can see Zapdos, Mew, Arcticuno and Moltres in Gen1. I haven't played the sequels but if you can name them all, feel free to comment some.

# In[ ]:


legendaries = df[df.Legendary == True].sort_values("Total", ascending = False).sort_values('Total')
strongest_legendaries = list(legendaries.Name)[-1:-4:-1]
print("The strongest legendaries are", ', '.join(strongest_legendaries))


# In[ ]:


labels = np.array([
    "HP",
    "Attack",
    "Defense",
    "Special Attack",
    "Special Defense",
    "Speed"
])

columns = np.array([5,6,7,8,9,10])

stats = df.iloc[4,columns].values

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=stats,
      theta=labels,
      fill='toself',
      name='Charmander'
))

stats = df.iloc[9, columns].values

fig.add_trace(go.Scatterpolar(
      r=stats,
      theta=labels,
      fill='toself',
      name='Squirtle'
))

stats = df.iloc[0, columns].values

fig.add_trace(go.Scatterpolar(
      r=stats,
      theta=labels,
      fill='toself',
      name='Bulbasaur'
))

fig.update_layout(
    autosize=False,
    width=500,
    height=500,)

fig.show()


# ## Looks like Bulbasaur might be the best starter pokemon, but I remember Squirtle being the recomended one even though he is the weakest one in the comparison. This was because of the class sytem in pokemon which is being overlooked in the comparision. Squirtle is stronger against the enemies in the first few gyms we encounter and hence is the better choice as class advantages boost the power of the pokemon. 
# 
# ## Let's make a some functions which would help us get results based on class as well.

# In[ ]:


class_disadvantages = {
    'Grass': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug'],
    'Rock': ['Water', 'Grass', 'Fighting', 'Ground', 'Steel'],
    'Ice':['Fire', 'Fighting', 'Rock', 'Steel'],
    'Dragon': ['Ice', 'Dragon', 'Fairy'],
    'Dark': ['Fighting', 'Bug', 'Fairy'],
    'Psychic': ['Bug', 'Ghost', 'Dark'],
    'Bug': ['Fire', 'Flying', 'Rock'],
    'Flying': ['Electric', 'Ice', 'Rock'],
    'Steel': ['Fire', 'Fighting', 'Ground'],
    'Fire': ['Water', 'Ground', 'Rock'],
    'Fighting': ['Flying', 'Psychic', 'Fairy'],
    'Ground': ['Water', 'Grass', 'Ice'],
    'Ghost': ['Ghost', 'Dark'],
    'Poison': ['Ground', 'Psychic'],
    'Water': ['Electric', 'Grass'],
    'Fairy': ['Poison', 'Steel'],
    'Electric': ['Ground'],
    'Normal': ['Fighting'],
    'None': []
}


# In[ ]:


def opponents(name, legendary_and_mega = False):
    """
    Picks most class advantage pokemon to defeat given pokemon and sorts them according to Total power and puts them in a dataframe object

        Parameters
        ----------
        name : str
            name of the pokemon for which opponents are to be found
        
        legendary_and_mega : bool, optional
            if legendaries and mega evolutions must be added or no, by default False

        Returns
        -------
        dataframe of pokemon which would be most effective to defeat given pokemon sorted in descending order of Total power
    """
    type1 = df[df.Name == name]['Type 1'].to_string().split(' ')[-1]
    type2 = df[df.Name == name]['Type 2'].to_string().split(' ')[-1]
    strong = set(class_disadvantages[type1] + class_disadvantages[type2])
    strong = list(strong)
    if legendary_and_mega:
        opponents = df[(df["Type 1"].isin(strong)) | (df["Type 2"].isin(strong))]
        best_opponents = opponents.sort_values(["Total", "HP"], ascending = False)[:10].Name
    else:
        opponents = df[(df["Type 1"].isin(strong)) | (df["Type 2"].isin(strong))][~df.Legendary]
        best_opponents = opponents.sort_values(["Total", "HP"], ascending = False)[~df.Name.str.contains("Mega")]
    return best_opponents

    
def generation_picker(data, generations):
    """
    Returns pokemon from the given generations only

        Parameters
        ----------
        data : pandas.core.series.Series
            data which is to be filtered
        
        generations : list of int
            generation/s needed from the data

        Returns
        -------
        subset of data with pokemon only from the given list of generation/s
    """
    return data[data.Generation.isin(generations)]
    
def choose_from_team(opponent, team, legendary_and_mega = False):
    """
    Displays images of pokemon/s from our team best suited to defeat the given opponent

        Parameters
        ----------
        opponent : str
            name of the pokemon for which the opponent is to selected from the team
            
        team : list of str
            names of pokemon in the team as strings
        
        legendary_and_mega : bool, optional
            if legendaries and mega evolutions must be added or no, by default False

        Returns
        -------
        images of pokemon/s that would be most affective at defeating the opponents or a disclaimer if none is found
    """
    weakness = opponents(opponent, legendary_and_mega = legendary_and_mega).Name.to_list()
    needmons = [pokemon for pokemon in team if pokemon in weakness]
    img_o = plt.imread(f'/kaggle/input/pokemon-images-and-types/images/images/{opponent.lower()}.png')
    fig = plt.imshow(img_o)
    plt.title(f'{opponent} is the opponent')
    plt.show(fig)
    i = 1
    if not len(needmons):
        return f"No pokemon is strong against the {opponent}!"
    else:
        print(f"You can take the following pokemon/s from your team to beat {opponent}:")
        for needmon in needmons:
            plt.subplot(len(needmons)//2+1,2,i)
            img = plt.imread(f'/kaggle/input/pokemon-images-and-types/images/images/{needmon.lower()}.png')
            plt.imshow(img)
            plt.title(needmon)
            i += 1


# In[ ]:


generation_picker(opponents('Mew'), [1])


# Gotta evolve that Scyther and Pinsir to beat Mew in Generation 1.

# In[ ]:


choose_from_team("Onix", ["Squirtle", "Charmander", "Bulbasaur"])


# ## Onix is the most powerful pokemon in the first gym. Squirtle and Bulbasaur are recommended! Sorry Charmander, even though you look cool, Squirtle is just better. Still most of us would have chosen Charmander anyway.

# In[ ]:


choose_from_team("Pikachu", ["Squirtle", "Charmander", "Bulbasaur"])


# ## Looks like no one can beat the adorable Pikachu!

# **<font size = 6>If you liked this notebook, don't forget to upvote it!</font>**
