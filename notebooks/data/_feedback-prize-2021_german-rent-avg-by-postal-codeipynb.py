#!/usr/bin/env python
# coding: utf-8

# ### In average, what are the cheapest areas (PLZ / Postal code) one can live in Germany? 🤑 🇩🇪 🏠 
# ### Can we find some good deals close to the big cities?

# <img src="https://media.giphy.com/media/S2x05XOP1kbrmiNIgL/giphy.gif">

# Aerial footage shot by a drone flying over the city of Göppingen. Video courtesy of infodesignerin.

# # 1 - Introduction

# ## 1.1 - Dataset Description

# The data was scraped from Immoscout24, the biggest real estate platform in Germany. Immoscout24 has listings for both rental properties and homes for sale, however, the data only contains offers for rental properties.
# 
# **Content**
# 
# The data set contains most of the important properties, such as living area size, the rent, both base rent as well as total rent (if applicable), the location (street and house number, if available, ZIP code and state), type of energy etc. It also has two variables containing longer free text descriptions: description with a text describing the offer and facilities describing all available facilities, newest renovation etc.

# ## 1.2 - Dataset Dictionary

# - **regio1:** Bundesland
# 
# - **serviceCharge:** Auxiliary costs such as electricty or internet [Euro]
# 
# - **heatingType:** Type of heating
# 
# - **telekomTvOffer:** Is payed TV included? If so, which offer?
# 
# - **telekomHybridUploadSpeed:** How fast is the hybrid inter upload speed?
# 
# - **newlyConst:** Is the building newly constructed?
# 
# - **balcony:** Does the object have a balcony?
# 
# - **picturecount:** How many pictures were uploaded to the listing?
# 
# - **pricetrend:** Price trend as calculated by Immoscout
# 
# - **telekomUploadSpeed:** How fast is the internet upload speed?
# 
# - **totalRent:** Total rent (usually a sum of base rent, service charge and heating cost)
# 
# - **yearConstructed:** Construction year
# 
# - **scoutId:** Immoscout Id
# 
# - **noParkSpaces:** Number of parking spaces
# 
# - **firingTypes:** Main energy sources, separated by colon
# 
# - **hasKitchen:** Has a kitchen
# 
# - **geo_bln:** Same as regio1
# 
# - **cellar:** Has a cellar
# 
# - **yearConstructedRange:** Binned construction year, 1 to 9
# 
# - **baseRent:** Base rent without electricity and heating
# 
# - **houseNumber:** House number
# 
# - **livingSpace:** Living space in sqm
# 
# - **geo_krs:** District, above ZIP code
# 
# - **condition:** Condition of the flat
# 
# - **interiorQual:** Interior quality
# 
# - **petsAllowed:** Are pets allowed, can be yes, no or negotiable
# 
# - **street:** Street name
# 
# - **streetPlain:** Street name (plain, different formating)
# 
# - **lift:** Is elevator available
# 
# - **baseRentRange:** Binned base rent, 1 to 9
# 
# - **typeOfFlat:** Type of flat
# 
# - **geo_plz:** ZIP code
# 
# - **noRooms:** Number of rooms
# 
# - **thermalChar:** Energy need in kWh/(m^2a), defines the energy efficiency class
# 
# - **floor:** Which floor is the flat on
# 
# - **numberOfFloors:** Number of floors in the building
# 
# - **noRoomsRange:** Binned number of rooms, 1 to 5
# 
# - **garden:** Has a garden
# 
# - **livingSpaceRange:** Binned living space, 1 to 7
# 
# - **regio2:** District or Kreis, same as geo krs
# 
# - **regio3:** City/town
# 
# - **description:** Free text description of the object
# 
# - **facilities:** Free text description about available facilities
# 
# - **heatingCosts:** Monthly heating costs in [Euro]
# 
# - **energyEfficiencyClass:** Energy efficiency class (based on binned thermalChar, deprecated since Feb 2020)
# 
# - **lastRefurbish:** Year of last renovation
# 
# - **electricityBasePrice:** Monthly base price for electricity in € (deprecated since Feb 2020)
# 
# - **electricityKwhPrice:** Electricity price per kwh (deprecated since Feb 2020)
# 
# - **date:** Time of scraping

# # 2 - Development

# Importing the necessary libraries:

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt


# Importing the dataframe:

# In[ ]:


db = pd.read_csv("../input/apartment-rental-offers-in-germany/immo_data.csv")


# Checking the first entries:

# In[ ]:


db.head()


# Dropping some data that pollutes the dataset. These entires are test entries with "1234567" as total price, along with other ones where an error ocurred during data collection and (I guess) the comma was misplaced. The identification process for these entires was removed from the notebook, but can be seen on previous versions of it.

# In[ ]:


db.drop(239533, inplace=True)
db.drop(109047, inplace=True)
db.drop(23490, inplace=True)
db.drop(79903, inplace=True)
db.drop(260259, inplace=True)
db.drop(8406, inplace=True)


# Regular database checks:

# In[ ]:


db.shape


# In[ ]:


db.isnull().sum()


# In[ ]:


db.info()


# Dataset correlation heatmap:

# In[ ]:


cor = db.corr()
cor.style.background_gradient(cmap='coolwarm')


# Importing the shapefile to the notebook

# In[ ]:


total_data = gpd.read_file("../input/germany-plz/plz-5stellig.shp")


# In[ ]:


total_data.head(10)


# In[ ]:


total_data.plot(figsize=(20,20))


# Getting the averages by 5 digit postal code (PLZ). Please note that the entires that do not appear in the average collection will have "zero" attributed to them further down the notebook:

# In[ ]:


plz_avg = db.groupby(["geo_plz"])["totalRent"].mean()


# In[ ]:


plz_avg = pd.DataFrame(plz_avg)
plz_avg.reset_index(drop=False, inplace=True)


# In[ ]:


plz_avg = plz_avg[["geo_plz", "totalRent"]]

fill_na = pd.DataFrame()
fill_na["geo_plz"] = pd.DataFrame(total_data["plz"]).astype(int)


# In[ ]:


fill_na["is_in_results"] = fill_na["geo_plz"].isin(plz_avg["geo_plz"]).astype(int)


# In[ ]:


fill_na


# In[ ]:


to_be_filled = pd.DataFrame()
to_be_filled = fill_na.loc[fill_na['is_in_results'] == 0]


# In[ ]:


to_be_filled


# In[ ]:


to_be_filled["is_in_results"] = to_be_filled['is_in_results'].replace(0, np.NaN)


# In[ ]:


to_be_filled.columns = ["geo_plz", "totalRent"]


# In[ ]:


plz_avg = pd.concat([plz_avg, to_be_filled], axis=0)


# In[ ]:


plz_avg


# Appending the averages to the initial shapefile

# In[ ]:


total_data["plz"] = total_data.plz.astype(int)
plz_avg["geo_plz"] = plz_avg.geo_plz.astype(int)


# In[ ]:


total_data_final = total_data.merge(plz_avg, left_on="plz", right_on="geo_plz")
total_data_final = gpd.GeoDataFrame(total_data_final)


# In[ ]:


total_data_final


# In[ ]:


total_data_final.info()


# Plotting the final data:

# In[ ]:


total_data_final.plot(figsize=(20,20))


# Setting the parameters for plotting the map:

# In[ ]:


plt.rcParams["figure.figsize"] = (50,50)


# Final plot:

# In[ ]:


fig, ax = plt.subplots(1)

ax.axis('off')

ax.set_title("Average Rent in Euros by German 5-digit \"Postleitzahl\" (Zip Code)", fontdict={'fontsize': '50', 'fontweight': '10'})

total_data_final.plot(column="totalRent",
                      ax=ax,
                      legend=True,
                      scheme="natural_breaks",
                      k=20,
                      #cmap = 'cividis',
                      edgecolor = "0",
                      linewidth = 0.001,
                      missing_kwds={"color": "white",
                                    "edgecolor": "red",
                                    "hatch": "///",
                                    "label": "Missing values"});

ax.annotate("Souce: Kaggle Dataset", xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left', 
            verticalalignment='bottom', fontsize=25)

fig.savefig('map.eps', format='eps')
fig.savefig('map2.svg', format='svg')


# In[ ]:


total_data_final.explore()


# In order to see the full map in vector format, feel free to enter [my Google Drive folder](https://drive.google.com/drive/folders/1OejsL2djv4bDr4JgxlePzff90hW9tEQx?usp=sharing) and download the final plot.

# Credit for the shapefile:
# https://www.suche-postleitzahl.org/

# **PS: This is a work in progress.**👷
# Further documentation and plots coming soon! 
