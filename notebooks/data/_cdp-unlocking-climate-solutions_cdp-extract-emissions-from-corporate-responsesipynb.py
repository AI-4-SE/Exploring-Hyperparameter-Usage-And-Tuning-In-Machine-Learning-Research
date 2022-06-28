#!/usr/bin/env python
# coding: utf-8

# # CDP: Extract Emissions from Corporate Responses
# 
# In this notebook, extract CO2 emission data from corporate responses.  
# There are various questionnaires in CDP that relate to CO2 emission. I show the relation of these and extract the most suitable data.
# 
# 
# ![top_ima](https://i.imgur.com/lanKdmV.png)
# 
# **I hope this notebook helps your analysis and if it realizes, please upvote!**
# 
# 
# ## Questionnaires that relate to emissions
# 
# * C4: Targets and performance
# * C5: Emissions methodology
# * C6: Emissions data
# 
# The above questionnaires all require to answer Scope1/2 emissions. Which is most suitable?  
# 
# **In short, C6 is the easiest to use.**
# 
# I explain why is it and the difference between C4 & C5.
# 
# 
# ## C6: Emissions data
# 
# "C6: Emissions data" mainly asks Scope1, Scope2, and Scope3 emissions.  
# 
# The following picture is from `2019_Climate_Change_Questionnarie.pdf`. The ~2019 questionnaire document is most understandable because the response flow chart picture is attached.
# 
# ![questionnaire_c6](https://i.imgur.com/WzqzUkR.png)
# 
# If you are not very familiar with Scope, the following are short commentaries.
# 
# * Scope1: The emissions from the company's own fuels.
# * Scope2: The emissions from the supplied energy (ex. electric power).
#   * location based: Based on the grid-average emission factor data.
#   * market based; Based on the procurement source grid factor data. If you purchase renewable energy, you can reflect it.
# * Scope3: The emissions from the overall supply chain.
#   * There are 15 categories of relations in the company's supply chain and report CO2e each of these.
# 
# The method to calculate GHG emission is defined at [GHG protocal](https://ghgprotocol.org/).
# 
# 
# ### Extract each Scope emissions
# 
# Let's extract the emissions data from C6 responses.
# 

# In[ ]:


import os
import pandas as pd
import numpy as np
import altair as alt
import re
import json


# In[ ]:


RESPONSE_ROOT = "../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses"
YEARS = (2018, 2019, 2020)
cl_dfs = {}

for year in YEARS:
    kind = "Climate Change"
    file_name = "{}_Full_{}_Dataset.csv".format(year, kind.replace(" ", "_"))
    path = "{}/{}/{}".format(RESPONSE_ROOT, kind, file_name)
    df = pd.read_csv(path)
    cl_dfs[year] = df


# The emission data is asked at C6.1, C6.3 and C6.5.  
# To extract the data, we have to understand the structure of each questionnaire.
# 
# **`question_number` C6.1**
# 
# `column_number` = 1 and `row_number` = 1 is our target.
# 
# ![c6.1](https://i.imgur.com/Bk5O0cX.png)
# 
# 
# **`question_number` C6.3**
# 
# `column_number` = 1 (location-based), 2 (market-based)  and `row_number` = 1 is our target.
# 
# 
# ![c6.3](https://i.imgur.com/ASFPDuK.png)
# 
# 
# **`question_number` C6.5**
# 
# `column_number` = 2 is our target. There are various categories, for that reason we have to sum up these to get Scope3 data.
# 
# ![c6.5](https://i.imgur.com/AQ6E8rV.png)
# 
# 
# Additionally, you can confirm the response rate of each question from [this notebook](https://www.kaggle.com/takahirokubo0/overview-of-corporations-data-of-cdp).

# In[ ]:


def extract_c6_emissions(year_df):
    """
    Extract Scope1, Scope2 and Scope3 emissions from C6.
    """
    structure = {
        "C6.1": {
            "column_name": "Scope1",
            "column_number": 1,
            "row_number": 1
        },
        "C6.3": {
            "column_name": ["Scope2-location", "Scope2-market"],
            "column_number": [1, 2],
            "row_number": 1
        },
        "C6.5": {
            "column_name": ["Scope3"],
            "column_number": 2
        }
    }
    
    items = ["account_number", "organization", "survey_year",
             "question_number", "column_number", "row_number",
             "table_columns_unique_reference", "response_value"]
    
    c6_emissions = []
    for target_number in structure:
        location = structure[target_number]
        df = year_df[year_df["question_number"] == target_number]
        
        # Select columns
        columns = location["column_number"]
        columns = columns if isinstance(columns, list) else [columns]
        for i, c in enumerate(columns):
            name = location["column_name"]
            name = name if isinstance(name, str) else name[i]
            selected = df[df["column_number"] == c]
            selected = selected[items]
            
            # Filter by rows
            if "row_number" in location:
                r = location["row_number"]
                selected = selected[selected["row_number"] == r]
            
            # Preprocess response value
            selected["response_value"] = pd.to_numeric(selected["response_value"], errors="coerce")
            selected = selected.dropna(subset=["response_value"])
            selected["scope"] = pd.Series([name] * len(selected), index=selected.index)
            c6_emissions.append(selected)
        
    c6_emissions = pd.concat(c6_emissions)
    items.append("scope")
    items.remove("row_number")
    c6_emissions = c6_emissions.groupby(items).sum().reset_index()
    
    return c6_emissions

c6_emissions_2020 = extract_c6_emissions(cl_dfs[2020])


# In[ ]:


c6_emissions_2020.head(5)


# In[ ]:


def test_emissions(year_df, emissions):
    """
    Test emissions values
    """
    
    import random
    master = year_df[year_df["question_number"].isin(["C6.1", "C6.3", "C6.5"])]

    def get_value(series):
        return float(series.tolist()[0])
    
    # C6.1
    have_scope1 = emissions[emissions["scope"] == "Scope1"]
    account_number = random.choice(have_scope1["account_number"].unique().tolist())
    df = have_scope1[have_scope1["account_number"] == account_number]
    c6_1 = df[(df["question_number"] == "C6.1") & (df["row_number"] == 1) & (df["column_number"] == 1)]
    em = emissions[(emissions["account_number"] == account_number) & (emissions["scope"] == "Scope1")]    
    assert len(c6_1) > 0
    assert len(em) > 0
    assert get_value(c6_1["response_value"]) == get_value(em["response_value"])

    # C6.3
    for i, s in enumerate(["Scope2-location", "Scope2-market"]):
        have_scope2 = emissions[emissions["scope"] == s]
        account_number = random.choice(have_scope2["account_number"].unique().tolist())
        df = master[master["account_number"] == account_number]

        c6_3 = df[(df["question_number"] == "C6.3") & (df["row_number"] == 1) & (df["column_number"] == i + 1)]
        em = emissions[(emissions["account_number"] == account_number) & (emissions["scope"] == s)]
        assert len(c6_3) > 0
        assert len(em) > 0
        assert get_value(c6_3["response_value"]) == get_value(em["response_value"])
    
    # C6.5
    have_scope3 = emissions[emissions["scope"] == "Scope3"]
    account_number = random.choice(have_scope3["account_number"].unique().tolist())
    df = master[master["account_number"] == account_number]

    c6_5 = df[(df["question_number"] == "C6.5") & (df["column_number"] == 2)]
    assert sum(c6_5["table_columns_unique_reference"].apply(lambda x: x.endswith("Metric tonnes CO2e"))) == len(c6_5)
    c6_5 = sum(c6_5["response_value"].dropna().astype(float))
    em = emissions[(emissions["account_number"] == account_number) & (emissions["scope"] == "Scope3")]
    em = sum(em["response_value"].astype(float))
    assert c6_5 == em
    
    return True


for i in range(10):
    test_emissions(cl_dfs[2020], c6_emissions_2020)


# Here is the simple visualization.

# In[ ]:


alt.Chart(
    c6_emissions_2020\
    .groupby(["organization", "scope"])\
    .sum()["response_value"]\
    .reset_index()\
    .sort_values(by="response_value", ascending=False).head(30)
).mark_bar().encode(
    x="response_value:Q",
    y=alt.Y("organization", sort=alt.EncodingSortField(field="response_value", order="descending")),
    color="scope"
)


# ## C5: Emissions methodology
# 
# "C5: Emissions methodology" is questionnaire for "methodology" and it asks **"base year"** Scope1 and Scope2 .
# 
# ![c5.jpg](https://i.imgur.com/9gNntvE.png)
# 
# 
# Sadly, response of C5 is missiong in this competition data.  
# For that reason, you can't use C5.
# 

# In[ ]:


for year in cl_dfs:
    df = cl_dfs[year]
    c5_1 = df[df["question_number"] == "C5.1"]
    print("Number of responses for C5.1 at {} is {}.".format(year, len(c5_1)))


# (It was too late to notice this fact before I started implementation).  
# The following code will be helpful someday we can access C5.

# In[ ]:


def extract_c5_emissions(year_df, emissions="emissions"):
    """
    Extract base year's Scope1, Scope2 emissions from C5.
    """
    structure = {
        "C5.1": {
            "columns": ["base_year_begin", "base_year_end", "emissions", None],
            "rows": ["Scope1", "Scope2-location", "Scope2-market"]
        }
    }
    
    items = ["account_number", "organization", "survey_year",
             "question_number", "column_number", "row_number",
             "table_columns_unique_reference", "response_value"]
    
    c5_emissions = []
    for target_number in structure:
        location = structure[target_number]
        df = year_df[year_df["question_number"] == target_number]

        print(len(df))

        # Select columns
        bases = []
        for i, c in enumerate(location["columns"]):
            if c is None:
                continue
            selected = df[df["column_number"] == i]
            selected = selected[items]
            selected.rename(columns={"response_value": c}, inplace=True)
            if len(bases) == 0:
                bases.append(selected)
            else:
                bases.append(selected[c])
        
        bases = pd.concat(bases, axis=1)
        # Select rows
        rows = []
        for j, r in enumerate(location["rows"]):
            if r is None:
                continue
            rows.append(j)
        bases = bases[bases["row_number"].isin(rows)]
        scope = bases["row_number"].apply(lambda i: location["rows"][i])
        bases.insert(len(bases.columns), "scope", scope)


        # Preprocess response value
        bases[emissions] = pd.to_numeric(bases[emissions], errors="coerce")
        bases = bases.dropna(subset=[c for c in location["columns"] if c is not None])
        c5_emissions.append(bases)

    c5_emissions = pd.concat(c5_emissions)
    
    return c5_emissions

extract_c5_emissions(cl_dfs[2020])


# ## C4: Targets and performance
# 
# 
# "C4: Targets and performance" asks the reduction plan and its progress than emissions itself.
# It relates TCFD "Metrics and Targets".
# 
# ![c4](https://i.imgur.com/TOYUGP6.png)
# 
# The C4 questionnarie branch off "C4.1a" and "C4.2b" depends on unit or target.
# 
# * C4.1a: Reduce the actual emissions in a future year (="Absolute target").
# * C4.2b: Calculate reduction based on the normalized value by business metric (="Intensity target").
# 
# How to set the target (C4) and how much amount of CO2 is emitted (C6) is another story.
# For that reason, to use C6 is suitable if you want to know actual emissions.
# 
# The structure of C4 is the following.
# 
# ![c4_structure](https://i.imgur.com/DI2CebL.png)
# 
# Let's watch the distribution of kinds of targets.

# In[ ]:


def compare_c4_target(year_dfs):
    """
    Compare target kind
    """
    SCOPE_COLUMN = {
        2018: 2,
        2019: 2,
        2020: 4
    }
    items = ["account_number", "organization", "survey_year",
             "question_number", "column_number", "row_number",
             "table_columns_unique_reference", "response_value"]

    c4_responses = []
    for year in year_dfs:
        df = year_dfs[year]
        column = SCOPE_COLUMN[year]
        for q in ("C4.1a", "C4.1b"):
            responses = df[(df["question_number"] == q) & (df["column_number"] == column)][items]
            responses = responses.dropna(subset=["response_value"])
            responses.rename(columns={"response_value": "scope"}, inplace=True)
            responses["survey_year"] = str(year)
            c4_responses.append(responses)
        
    c4_responses = pd.concat(c4_responses).reset_index(drop=True)
    return c4_responses


c4_responses = compare_c4_target(cl_dfs)


# In[ ]:


def visualize_c4(c4_responses, threshold=10):
    """
    Visualize Target kinds.
    """
    r = c4_responses\
            .groupby(["scope", "survey_year"])\
            .size()\
            .reset_index(name="count")
    
    r = r[r["count"] > threshold]
    return alt.Chart(r).mark_bar().encode(
        x="count:Q",
        y=alt.Y("scope", sort=alt.EncodingSortField(field="count", order="descending")),
        color="survey_year"
    )
visualize_c4(c4_responses)


# The most of targets are mixed Scope 1 and Scope 2.  
# It seems to be difficult to get the pure value of each Scope.
# 
# 
# ## Back to C6 and Visualize Emissions
# 
# At last, we back to C6 and visualize its data.
# 
# Let's extract corporate attributes (ticker etc) from disclosure data.

# In[ ]:


DISCLOSURE_ROOT = "../input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing"
YEARS = (2018, 2019, 2020)
cl_ddfs = {}

for year in YEARS:
    kind = "Climate Change"
    file_name = "{}_Corporates_Disclosing_to_CDP_{}.csv".format(year, kind.replace(" ", "_"))
    path = "{}/{}/{}".format(DISCLOSURE_ROOT, kind, file_name)
    df = pd.read_csv(path)
    cl_ddfs[year] = df


# In[ ]:


def extract_master(year_df):
    """
    Extract corporate attribute data.
    """
    items = [
        "account_number",
        "primary_ticker",
        "organization",
        "survey_year",
        "country",
        "region",
        "authority_types",
        "activities",
        "sectors",
        "industries",
        "primary_activity",
        "primary_sector",
        "primary_industry",
        "primary_questionnaire_sector",
    ]
    df = year_df[items]
    df.dropna(subset=["account_number", "primary_ticker", "survey_year"], inplace=True)
    return df


extract_master(cld_dfs[2020]).head(5)


# Now join the disclosure and emissions.

# In[ ]:


def make_emissions_data(year_dfs, year_ddfs):
    """
    Make emissions data by using C6 and disclosure data
    """
    
    emissions = []
    for year in year_dfs:
        c6 = extract_c6_emissions(year_dfs[year])
        c6.rename(columns={"response_value": "emissions"}, inplace=True)
        master = extract_master(year_ddfs[year])
        df = c6.merge(master, how="inner", on=["account_number", "survey_year"], suffixes=("_emission", None))
        df["survey_year"] = str(year)
        emissions.append(df)
    
    emissions = pd.concat(emissions)
    return emissions

emissions = make_emissions_data(cl_dfs, cl_ddfs)


# In[ ]:


emissions.head(5)


# Let's visualize emissions.  
# You can filter the data by click.

# In[ ]:


def visualize_emissions(emissions, height=300, left_width=100, right_width=500):
    """
    Visualize emissions intaractively
    Left: yearly emissions
    Right: emissions in each sector
    """
    
    items = ["survey_year", "primary_industry", "scope", "emissions"]
    source = emissions[items]\
                .groupby(items[:-1])\
                .sum()\
                .reset_index()
        
    selector = alt.selection_single(empty="all", fields=["scope"])
    base = alt.Chart(source).properties(
                height=height
            ).add_selection(selector)
    
    
    left = base.mark_bar().encode(
                x="survey_year",
                y="emissions:Q",
                color=alt.condition(selector, "scope", alt.value("lightgray"))
            ).properties(width=left_width)
    
    
    right = base.mark_bar().encode(
                x="survey_year",
                y=alt.Y("emissions:Q", stack="normalize"),
                color="primary_industry",
            ).transform_filter(
                selector
            ).properties(width=right_width)
    
    return left | right


visualize_emissions(emissions)


# We can confirm "Food, beverage" decrease its emissions but "Manufacturing" increases.  
# Is this means "Food, beverage" is eager to reduce emissions, and "Manufacturing" is lazy to climate change?
# 
# We need more surveys to unravel observation!
