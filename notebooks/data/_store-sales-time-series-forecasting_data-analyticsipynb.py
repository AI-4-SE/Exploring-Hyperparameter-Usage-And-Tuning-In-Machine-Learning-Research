#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import csv

# Stockmarket data

files = os.listdir('/kaggle/input/stock-market-data/stock_market_data/sp500/csv')

# Apple data
for file in files:
      if((file).casefold() == "aapl.csv"):
            apple_dir = ('/kaggle/input/stock-market-data/stock_market_data/sp500/csv/' +file)
            print("apple_dir")
apple_data = pd.read_csv(apple_dir)
print(apple_data)  






# Macroecnomic data
filename = '../input/interest-rates/index.csv'

#importing data & creating lists
with open(filename) as f:
	reader = csv.reader(f)
	header_row = next(reader)
	years, months, fed_target_rates, effective_fed_rates, real_gdps, unemployment_rates, inflation_rates = [], [], [], [], [], [], []
    
	print(header_row)
        
	for row in reader:
        
        # interest rates
		fed_target_rate = row[3]
		effective_fed_rate = row[6]
		fed_target_rates.append(fed_target_rate)
		effective_fed_rates.append(effective_fed_rate)        
        
        # real_gdp
		real_gdp = row[7]
		real_gdps.append(real_gdp)        
        
        # unemployment
		unemployment_rate = row[8]
		unemployment_rates.append(unemployment_rate)        
        
        # inflation
		inflation_rate = row[9]
		inflation_rates.append(inflation_rate)	
        
        # when exactly?
		year = row[0]
		month = row[1]
		years.append(year)
		months.append(month)

# M2 Money Supply
    
filename = '../input/real-m2-money-stock/M2REAL.csv'
m2_money_supply_list, m2_dates = [], []

with open(filename) as f:
	reader = csv.reader(f)
	header_row = next(reader)
    
	print(header_row)
    
	for row in reader:
		m2_money_supply = row[3]
		m2_money_supply_list.append(m2_money_supply)
		date = row[2]
		m2_dates.append(date)
        
print(m2_money_supply_list[:10])
print(m2_dates[:10])

# Bei beiden Macro-Daten sind die Sätze für jeden Tag. Umwandeln für den ersten des Monats?
        







