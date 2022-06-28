#!/usr/bin/env python
# coding: utf-8

# # What is ASHRAE Competition?

# In this competition, you’ll develop accurate models of metered building energy usage

# Data Download
# - kaggle competitions download -c ashrae-energy-prediction

# ## Data description

# - building_metadata.csv
#         - site_id
#         - building_id
#         - primary_use
#         - square_feet
#         - year_built
#         - floor_count
# - train.csv
#         - building_id
#         - meter
#         - timestamp
#         - meter_reading
# - test.csv
#         - row_id
#         - building_id
#         - meter
#         - timestamp
# - weather_train.csv & weather_test.csv
#         - site_id
#         - timestamp
#         - air_temperature
#         - cloud_coverage
#         - dew_temperature
#         - precip_depth_1_hr
#         - sea_level_pressure
#         - wind_direction
#         - wind_speed

# - building_id - Foreign key for the building metadata.
# - meter - The meter id code. Read as {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}. Not every building has all meter types.
# - timestamp - When the measurement was taken
# - meter_reading - The target variable. Energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error. UPDATE: as discussed here, the site 0 electric meter readings are in kBTU.
# - site_id - Foreign key for the weather files.
# - building_id - Foreign key for training.csv
# - primary_use - Indicator of the primary category of activities for the building based on EnergyStar property type definitions
#         - Banking/financial services
#         - Education
#         - Entertainment/public assembly
#         - Food sales and service
#         - Healthcare
#         - Lodging/residential
#         - Manufacturing/industrial
#         - Mixed use
#         - Office
#         - Parking
#         - Public services
#         - Religious worship
#         - Retail
#         - Technology/science
#         - Services
#         - Utility
#         - Warehouse/storage
#         - Other 
# - square_feet - Gross floor area of the building
# - year_built - Year building was opened
# - floor_count - Number of floors of the building
# - air_temperature - Degrees Celsius
# - cloud_coverage - Portion of the sky covered in clouds, in oktas
# - dew_temperature - Degrees Celsius
# - precip_depth_1_hr - Millimeters
# - sea_level_pressure - Millibar/hectopascals
# - wind_direction - Compass direction (0-360)
# - wind_speed - Meters per second

# # My solution

# - I used the data by feather format for fast loading
# => https://www.kaggle.com/corochann/ashrae-feather-format-for-fast-loading
# - This solution got 92th rank

# - ensemble five different model with different feature engineering
# - conduct weighted average ensemble by using leak data as validation data

# ## 1 : k-folds(k = 3) model

# - merge buidling and weather data into train data
# - separate timestamp into hour, mounth, weekday
# - fill and update the null value of weather data
# - drop "timestamp","sea_level_pressure", "wind_direction", "wind_speed","year_built","floor_count"
# - label encode 'primary use'

# I conducted k-folds model with k is 3 by using LightGBM
#  => https://www.kaggle.com/yunishi0716/k-folds-model

# ## 2 half and half model

# - merge buidling and weather data into train data
# - add new features extracting from timestamp as hour, weekday
# - based on calender, add new feature, is_holiday

# I choose two-folds LightGBM split half and half => https://www.kaggle.com/yunishi0716/half-and-half-model-ashre?scriptVersionId=28053730

# ## 3: Highway route4

# - remove weired data(t is reported in this discussion by @barnwellguy that All electricity meter is 0 until May 20 for site_id == 0)
# - fix time zone
# - add date featueres(hour, month, weekend, dayofweek)
# - fill the null values of weather by interpolation(linear, both limit direction)

# train the 3-folds LightGBM model by each meter type => https://www.kaggle.com/yunishi0716/3-folds-by-each-meter-type?scriptVersionId=28056268

# ## 4: EDA model(3-folds LightGMB)

# - fill and update the null values of weather data as I did (1)k-folds model
# - add new features, Age(how long have been taken the building from the last time construction)
# - delete weired data as 3
# - delete train data with that meter type is electricity and meter reading is 0
# - add mean, median and std of meter reading groupby buidling_id, month, day of week and meter
# - drop year_built
# - drop columns which has more than 0.9 correlation to other columns

# Training model is 3-folds LightGMB => https://www.kaggle.com/kulkarnivishwanath/ashrae-great-energy-predictor-iii-eda-model

# ## 5: data clean up model

# - drop row satisfying
#     - all electricity with 0 meter reading
#     - more than 48 hours run of steam and hot-water zero-reading(except to core summer month)
#     - more than 48 hours run of chilledwater zero-reading(except to ccur simultaneously at the start and end of the year)
# - drop electrical readings from the first 141 days of the data for site 0
# - drop building 1099
# - fill null values of weather data by interpolation
# - fix timestamp

# Training model is LightGBM => https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks/notebook

# # Weighted average ensemble and leak data validation

# Kernel => https://www.kaggle.com/yunishi0716/best-weight-searching3?scriptVersionId=25352432
