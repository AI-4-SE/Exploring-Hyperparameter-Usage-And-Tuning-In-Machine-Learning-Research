#!/usr/bin/env python
# coding: utf-8

# # Barcelona Accidents 2017 (EDA)
#     This case study aims to anlayze Barcelona's 2017 accidents dataset, published by the Open Data BCN, exploring the correlation between accident location (i.e district, neighborhood), and its consequences (i.e Victims, Serious injuries, etc..)

# ### Table of contents
# * <a href="#OneZero">Accidents by district %</a>
#     * <a href="#OneOne">Accidents by Neighborhood %</a>
# * <a href="#TwoZero">Serious Injuries Occurence by District %</a>
#     * <a href="#TwoOne">Serious Injuries Occurence by Neighborhood %</a>
# * <a href="#ThreeZero">Total Injuries by District %</a>
#     * <a href="#ThreeOne">Total Injuries by Neighborhood %</a>
# * <a href="#FourZero">Victim Occurence by District %</a>
#     * <a href="#FourOne">Victim Occurence by Neighborhood %</a>
# * <a href="#Summary">Summary</a>
# * <a href="#Conclusion">Conclusion</a>

# ![barca.jpg](attachment:barca.jpg)

# <hr style="border:1px solid gray"> </hr>

# In[ ]:


import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets import Layout
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


Acc_Data = pd.read_csv('../input/barcelona-data-sets/accidents_2017.csv')


# In[ ]:


Acc_Data.head()


# In[ ]:


list(Acc_Data)


# <hr style="border:1px solid gray"> </hr>

# <a id='OneZero'></a>
# ### 1) What is the District with the highest/lowest accident percentage?

# In[ ]:


District = Acc_Data["District Name"].value_counts()
District_D = pd.DataFrame(District)
District_D


# In[ ]:


District_D = District_D / sum(District_D["District Name"])
District_D = District_D * 100
District_D.rename(columns = {"District Name":"Accidents by district %"}, inplace = True)
District_D


# ### 1) Accidents by district %
# <ul>
#     <li>Accidents occured in Unknown districts is considered a bias in our analysis, but it comes in an acceptable ratio (0.26%)</li>
#     <li>Barcelona's Plaça de Catalunya, Eixample, records the highest accidents by district percentage (29.3%).</li>
#     <li> All other districts come with low accidents percentage, compared to Eixample.</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='OneOne'></a>
# ### 1.1) What about Eixample Neighborhoods?

# In[ ]:


Acc_Data.head()


# In[ ]:


Eixample_D = Acc_Data.loc[Acc_Data["District Name"] == "Eixample"]
Eixample_D.head()


# In[ ]:


Eixample_Neighborhood = Eixample_D["Neighborhood Name"].value_counts()
Eixample_Neighborhood_D = pd.DataFrame(Eixample_Neighborhood)
Eixample_Neighborhood_D


# In[ ]:


Eixample_Neighborhood_D = Eixample_Neighborhood_D / sum(Eixample_Neighborhood_D["Neighborhood Name"])
Eixample_Neighborhood_D = Eixample_Neighborhood_D * 100
Eixample_Neighborhood_D.rename(columns = {"Neighborhood Name":"Eixample Accidents by Neighborhood %"}, inplace = True)
Eixample_Neighborhood_D


# ### 1.1) Eixample Accidents by Neighborhood %
# <ul>
#     <li>la Dreta de l'Eixample, records the highest accidents percentage (38.5%) of the 29.3% (Eixample accidents %) of all Barcelona accidents.</li>
#     <li>Sant Antoni & Fort Pienc are considered the safest neighborhoods in Eixample.</li>
#     <li>All other neighborhoods come with an average accidents percentage, compared to la Dreta de l'Eixample.</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='TwoZero'></a>
# ### 2) Now, about serious injuries, what is the district with the most/least serious injuries occurences?

# In[ ]:


Acc_Data.head()


# In[ ]:


SerInj_D = Acc_Data.loc[Acc_Data["Serious injuries"] != 0]
SerInj_D.head()


# In[ ]:


District_SerInj = SerInj_D["District Name"].value_counts()
District_SerInj_D = pd.DataFrame(District_SerInj)
District_SerInj_D


# In[ ]:


District_SerInj_D = District_SerInj_D / sum(District_SerInj_D["District Name"])
District_SerInj_D = District_SerInj_D * 100
District_SerInj_D.rename(columns = {"District Name" : "Serious Injuries Occurence by District %"}, inplace = True)
District_SerInj_D


# ### 2) Serious Injuries Occurence by District %
# <ul>
#     <li>Unknown districts bias comes with an acceptable percentage (0.45%)</li>
#     <li>As expected, Barcelona's Plaça de Catalunya comes in first place considering serious injuries occurences with 25.56%</li>
#     <li>All other districts, except for Sant Mari, have low serious injuries occurence percentage</li>
#     <li>Sant Andreu comes the third least accident rate, also the third least serious injuries occurence rate. A relativly safe district.</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='TwoOne'></a>
# ### 2.1) What about Eixample neighborhoods, does la Dreta de l'Eixample also comes in the first place considering serious injuries?

# In[ ]:


SerInj_D.head()


# In[ ]:


Eixample_SerInj_D = SerInj_D.loc[SerInj_D["District Name"] == "Eixample"]
Eixample_SerInj_D.head()


# In[ ]:


Eixample_Neighborhood_SerInj = Eixample_SerInj_D["Neighborhood Name"].value_counts()
Eixample_Neighborhood_SerInj_D = pd.DataFrame(Eixample_Neighborhood_SerInj)
Eixample_Neighborhood_SerInj_D


# In[ ]:


Eixample_Neighborhood_SerInj_D = Eixample_Neighborhood_SerInj_D / sum(Eixample_Neighborhood_SerInj_D["Neighborhood Name"])
Eixample_Neighborhood_SerInj_D = Eixample_Neighborhood_SerInj_D * 100
Eixample_Neighborhood_SerInj_D.rename(columns = {"Neighborhood Name" : "Eixample Serious Injuries Occurence by Neighborhood %"}, inplace = True)
Eixample_Neighborhood_SerInj_D


# ### 2.1) Eixample Serious Injuries by Occurence Neighborhood %
# <ul>
#     <li>As expected too, la Dreta de l'Eixample neighborhood records the highest Serious injuries occurence percentage (45.6%) of the 25.56% (Eixample serious injuries occurence %) of all Barcelona serious injuries occurences.</li>
#     <li>The order of Eixample neighborhoods considering accidents is the same as their order considering serious injuries occurence, except for that el Fort pienc has higher serious injuries % than la Sagrada Familia.</li>
#     <li>Sant Antoni, la Sagrada Familia, and el Fort Pienc are the least considering serious injuries occurence.</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='ThreeZero'></a>
# ### 3) Talking about injuries, what about the total number of injuries (mild + serious)?

# In[ ]:


Acc_Data.head()


# In[ ]:


Total_Injuries = Acc_Data["Mild injuries"] + Acc_Data["Serious injuries"]
Acc_w_TotInj_D = Acc_Data.assign(TotalInjuries = Total_Injuries)
Acc_w_TotInj_D.rename(columns = {"TotalInjuries" : "Total Injuries"}, inplace = True)
Acc_w_TotInj_D


# In[ ]:


District_TotInj_D = pd.pivot_table(data = Acc_w_TotInj_D, index = ["District Name"], values = "Total Injuries", aggfunc="sum")
District_TotInj_D.sort_values(by = ["Total Injuries"], ascending = False, inplace = True)
District_TotInj_D


# In[ ]:


District_TotInj_D = District_TotInj_D / sum(District_TotInj_D["Total Injuries"])
District_TotInj_D = District_TotInj_D * 100
District_TotInj_D.rename(columns = {"Total Injuries" : "Total Injuries by District %"}, inplace = True)
District_TotInj_D


# ### 3) Total Injuries by District %
# <ul>
#     <li>Unknown district bias comes in an acceptable percentage (0.33%)</li>
#     <li>How dangerous is Barcelona's Plaça de Catalunya!! Eixample comes the first in everything considering accidents.</li>
#     <li>Most of other districts have low number of total injuries, compared to Eixample</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='ThreeOne'></a>
# ### 3.1) What about Eixample neighborhoods, does la Dreta de l'Eixample also that dangerous considering total injuries?

# In[ ]:


Acc_w_TotInj_D.head()


# In[ ]:


Eixample_Acc_TotInj_D = Acc_w_TotInj_D.loc[Acc_w_TotInj_D["District Name"] == "Eixample"]
Eixample_Acc_TotInj_D.head()


# In[ ]:


Eixample_Neighborhood_TotInj_D = pd.pivot_table(data = Eixample_Acc_TotInj_D, index = ["Neighborhood Name"], values = "Total Injuries", aggfunc="sum")
Eixample_Neighborhood_TotInj_D.sort_values(by = ["Total Injuries"], ascending = False, inplace = True)
Eixample_Neighborhood_TotInj_D


# In[ ]:


Eixample_Neighborhood_TotInj_D = Eixample_Neighborhood_TotInj_D / sum(Eixample_Neighborhood_TotInj_D["Total Injuries"])
Eixample_Neighborhood_TotInj_D = Eixample_Neighborhood_TotInj_D * 100
Eixample_Neighborhood_TotInj_D.rename(columns = {"Total Injuries" : "Eixample Total Injuries by Neighborhood %"}, inplace = True)
Eixample_Neighborhood_TotInj_D


# ### 3.1) Eixample Total Injuries by Neighborhood %
# <ul>
#     <li>As always, la Dreta de l'Eixample neighborhood has the highest total injuries percentage (38.44%) of the 29.23% (Eixample total injuries %) of all Barcelona total injuries.</li>
#     <li>Considiring the past accident statistics (No. of accidents, No. of serious injuries), la Sagrada unexpectedly came in the third place considring Eixample neighborhood total injuries</li>
#     <li>Sant Antoni, and el Fort Pienc are the safest considering total injuries.</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='FourZero'></a>
# ### 4) Lets, sadly, have a look on victims, what is the district with the highest victim/s occurences?

# In[ ]:


Acc_Data.head()


# In[ ]:


Victims_D = Acc_Data.loc[Acc_Data["Victims"] != 0]
Victims_D.head()


# In[ ]:


District_Victims = Victims_D["District Name"].value_counts()
District_Victims_D = pd.DataFrame(District_Victims)
District_Victims_D


# In[ ]:


District_Victims_D = District_Victims_D / sum(District_Victims_D["District Name"])
District_Victims_D = District_Victims_D * 100
District_Victims_D.rename(columns = {"District Name" : "Victim Occurence by District %"}, inplace = True)
District_Victims_D


# ### 4) Victim Occurence by District %
# <ul>
#     <li>Unknown districs bias comes in an acceptable percentage (0.28%).</li>
#     <li>Again, Eixample shows off how dangerous it is with a 30.4% Victim Occurence.</li>
#     <li>Gracia, Nou Barris, Ciutat vella, Sant Andreu, Horta-Guinardo, and Les Corts are the safest with least victims.</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='FourOne'></a>
# ### 4.1) What about Eixample neighborhoods, will la Dreta de l'Eixample come in the first place again considering victim occurence?

# In[ ]:


Victims_D.head()


# In[ ]:


Eixample_Victims_D = Victims_D.loc[Victims_D["District Name"] == "Eixample"]
Eixample_Victims_D.head()


# In[ ]:


Eixample_Neighborhood_Victims = Eixample_Victims_D["Neighborhood Name"].value_counts()
Eixample_Neighborhood_Victims_D = pd.DataFrame(Eixample_Neighborhood_Victims)
Eixample_Neighborhood_Victims_D


# In[ ]:


Eixample_Neighborhood_Victims_D = Eixample_Neighborhood_Victims_D / sum(Eixample_Neighborhood_Victims_D["Neighborhood Name"])
Eixample_Neighborhood_Victims_D = Eixample_Neighborhood_Victims_D * 100
Eixample_Neighborhood_Victims_D.rename(columns = {"Neighborhood Name" : "Eixample Victim Occurence by Neighborhood"}, inplace = True)
Eixample_Neighborhood_Victims_D


# ### 4.1) Eixample Victim Occurence by Neighborhood %
# <ul>
#     <li>Again and again, la Dreta de l'Eixample neighborhood has the highest victim occurence percentage (38.06%) of the 30.4% (Eixample victim occurence %) of all Barcelona victim occurences.</li>
#     <li>Sant Antoni, and el Fort Pienc are the safest considering victim occurences.</li>
# </ul>

# <hr style="border:1px solid gray"> </hr>

# <a id='Summary'></a>
# ### Summing all up

# In[ ]:


Districts_stats = pd.concat([District_D, District_SerInj_D, District_TotInj_D, District_Victims_D], axis = 1)
Districts_stats


# In[ ]:


Eixample_Neighborhood_stats = pd.concat([Eixample_Neighborhood_D, Eixample_Neighborhood_SerInj_D, Eixample_Neighborhood_TotInj_D, Eixample_Neighborhood_Victims_D], axis = 1)
Eixample_Neighborhood_stats


# In[ ]:


#Dropdown list
case = widgets.Dropdown(
    value = list(Districts_stats)[0],
    options = list(Districts_stats),
    description = 'Case',
    layout = Layout(width='35%', height='40px', display='flex')
    )


# In[ ]:


# Interactive Barchart update
def updateBar(case):
    ChosenCase = case
    fig = plt.figure(figsize = (20, 5))
    sns.set_style("whitegrid")
    sns.barplot(x = list(Districts_stats.index), y = Districts_stats[ChosenCase].values.tolist(), palette = 'Blues_r')
    plt.xlabel("Districts", fontsize = 20)
    plt.ylabel("Percentage %", fontsize = 20)
    plt.title(ChosenCase, fontsize = 25)
    plt.show()


# <hr style="border:1px solid gray"> </hr>

# <a id='Conclusion'></a>
# # Conclusion
# ### *By analysing each of the following (Accidents, Serious injury occurence, Total injuries, Victim occurences)*
# 
# 

# ### 1) Eixample is the most dangerous district in Barcelona
# <ul>
#     <li>Highest accident percentage (29.3%)</li>
#     <li>Highest serious injuries occurence percentage (25.56%)</li>
#     <li>Highest toal injuries percentage (29.23%)</li>
#     <li>Highest Victim occurence (30.4%)
# </ul>

# ###  2) Gracia is the safest district in Barcelona
# <ul>
#     <li>Lowest accidents percentage (5.14%)</li>
#     <li>Fourth lowest serious injury occurence percentage (4.93%)</li>
#     <li>Lowest total injury percentage (4.83%)</li>
#     <li>Lowest victim occurence percentage (5.11%)</li>
# </ul>
# *Nou Barris, Ciutat Vella, Sant Andreu, Horta-Guinardo, and Les Corts are also considered safe districts (in order after Gracia)*

# In[ ]:


widgets.interactive(updateBar, case = case)


# ## Diving deep into Eixample neighborhoods

# ### 3) la Dreta de l'Eixample is the most dangerous neighborhood in Eixample
# <ul>
#     <li>Highest accident percentage (38.53%)</li>
#     <li>Highest serious injuries occurence percentage (45.61%)</li>
#     <li>Highest toal injuries percentage (38.44%)</li>
#     <li>Highest Victim occurence (38.06%)</li>
# </ul>

# ### 4) Sant Antoni is the safest neighborhood in Eixample
# <ul>
#     <li>Lowest accidents percentage (7.69%)</li>
#     <li>Lowest serious injury occurence percentage (3.51%)</li>
#     <li>Lowest total injury percentage (7.92%)</li>
#     <li>Lowest victim occurence percentage (7.22%)</li>
# </ul>
# *el Fort Pienc is also considered a safe neighborhood (after Sant Antoni)*

# <hr style="border:1px solid gray"> </hr>
