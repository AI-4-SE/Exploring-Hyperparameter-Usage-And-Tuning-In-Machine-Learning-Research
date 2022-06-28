#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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


# ## Importing data and Indian GeoJSON

# GeoJSON is a special type of file which is made to represent simple geographical features, along with their non-spatial attributes. Here the features' polygonal coordinates relate with their actual coordinates on the globe, making it easier to plot entities likes districts and states if their coordinates are not provided already in the module. I have chosen the latest Indian map GeoJSON as my data pertains to India.

# In[ ]:


df_flights = pd.read_csv("../input/airtrafficcoordinatesindia/flight_data.csv")
df_cities = pd.read_csv("../input/india-cities-with-geolocations/city_locations.csv")
df_rainfall = pd.read_csv("../input/rainfall-data-from-1901-to-2017-for-india/Rainfall_Data_LL.csv")
df_corona = pd.read_csv("../input/latest-covid19-india-statewise-data/Latest Covid-19 India Status.csv")
india_geojson = json.load(open("../input/india-latest-geojson/india_states.geojson", "r"))


# In[ ]:


df_cities.head()


# In[ ]:


df_flights.head()


# While going through the data, I saw that the names in the csv and the JSON were not matching, so I changed the name values in DataFrame to be consistent with the ones in the JSON file.

# In[ ]:


df_corona.at[7, 'State/UTs'] = "Dadra and Nagar Haveli and Daman and Diu"
df_corona.at[13, 'State/UTs'] = "Jammu & Kashmir"
df_corona.at[31, 'State/UTs'] = "Telangana"
df_corona.head()


# # Map Styles

# ## Chloropleth

# Here we apply a colormap over our geographical features. The feature is filled with the hue of the respective value from the colormap. With this method we can only encode one attribute in our visualization.

# In[ ]:


fig = px.choropleth(
    df_corona,
    geojson=india_geojson,
    featureidkey='properties.ST_NM', 
    locations='State/UTs',
    color='Death Ratio',
    color_continuous_scale='Reds'
)

fig.update_geos(fitbounds="locations", visible=False)

fig.show()


# ## GeoScatter plot

# We can encode two attributes using a GeoScatter plot. We place a marker, usually a bubble, at the coordinates that we need to visualize. One of the attribute can be size of the bubble and the other can be the colour of the bubble. One drawback to this method is that the visualization can become messy if the bubble are too big, or become meaningless if the bubbles are too small to be legible. We must choose the right scale to fit to make the map meaningful.

# In[ ]:


fig = px.scatter_geo(
    df_corona,
    geojson=india_geojson,
    featureidkey='properties.ST_NM', 
    locations='State/UTs',
    size= df_corona['Total Cases'],
    color = 'Deaths',
    color_continuous_scale='Pinkyl',
)

fig.update_geos(fitbounds="locations")

fig.update_layout(
    title_text = 'Corona Active cases and deaths',
    geo = dict(
        showland = True,
        landcolor = 'rgb(00, 00, 00)',
        countrycolor = 'rgb(204, 204, 00)',
        scope = 'asia'
    ),
)

fig.show()


# ## Network map

# This data lets us know the connection between two points and the volume of flow between them. 

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon = df_flights['arr_lon'].append(df_flights['dep_lon']),
    lat = df_flights['arr_lat'].append(df_flights['dep_lat']),
    mode = 'markers',
    marker = dict(
    size = 4,
    color = 'rgb(255, 0, 0)',
    line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))

for i in range(len(df_flights)):
    fig.add_trace(
        go.Scattergeo(
            lon = [df_flights['arr_lon'][i], df_flights['dep_lon'][i]],
            lat = [df_flights['arr_lat'][i], df_flights['dep_lat'][i]],
            mode = 'lines',
            line = dict(width = 1.5,color = 'red'),
            opacity = float(df_flights['nb_flights'][i]) / float(df_flights['nb_flights'].max()),
        )
    )

fig.update_layout(
    title_text = 'Flights to and From India',
    showlegend = False,
    geo = dict(
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
    ),
)

fig.show()


# Let's zoom into Asia to get a better view of the map.

# In[ ]:


fig.update_layout(
    title_text = 'Flights to and From India',
    showlegend = False,
    geo = dict(
        showland = True,
        scope='asia',
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
    ),
)

fig.show()


# ## Density Heatmap

# This shows the density of the data onto the map. It is particularly useful for data which covers a lot of coordinate, like weather and population data.
# 
# You can run this notebook and change the year and month to see a completely different map and interpret your own results.

# In[ ]:


month = 'JUL'
year = 2017
rainfall_year = df_rainfall[df_rainfall.YEAR == year]


# In[ ]:


fig = px.density_mapbox(rainfall_year, lat='Latitude', lon='Longitude', z=month, radius=70,
                        center=dict(lat=22, lon=80), zoom=3,
                        mapbox_style="stamen-terrain", 
                        color_continuous_scale = 'Jet')

fig.update_layout(
    title_text = f'{month} {year} rainfall in mm'
)

fig.show()


# In[ ]:


fig = go.Figure()
fig = px.density_mapbox(df_cities, lat='Latitude', lon='Longitude', z='Population', radius=35,
                        center=dict(lat=22, lon=80), zoom=3,
                        mapbox_style="stamen-terrain", 
                        color_continuous_scale = 'Agsunset')

fig.update_layout(
    title_text = 'Population heatmap of india'
)

fig.show()


# # If you liked this notebook, don't forget to upvote it!
# 
