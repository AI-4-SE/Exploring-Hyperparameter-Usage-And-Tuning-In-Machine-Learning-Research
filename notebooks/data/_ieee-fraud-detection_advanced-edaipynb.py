#!/usr/bin/env python
# coding: utf-8

# <h2 style='background:#11489c; border:0; color:white'><center>Advanced Functional Exploratory Data Analysis</center></h2>
# 
# It is a study that focuses on advanced functionalized exploratory data analysis with a simple step-by-step explanation
# 
# <h2 style='background:#11489c; border:0; color:white'><center>What is Exploratory Data Analysis?</center></h2>
# 
# <a href="https://ibb.co/MMSWwR3"><img src="https://i.ibb.co/jgfB0L1/Screenshot-2022-01-05-030343.png" alt="Screenshot-2022-01-05-030343" border="0"></a>
# 
# Exploratory data analysis (EDA) is used by data scientists to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions.
# 
# EDA is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a provides a better understanding of data set variables and the relationships between them. It can also help determine if the statistical techniques you are considering for data analysis are appropriate. Originally developed by American mathematician John Tukey in the 1970s, EDA techniques continue to be a widely used method in the data discovery process today
# 
# It is a study that focuses on advanced functionalized exploratory data analysis with a simple step-by-step explanation
# 
# * <span style="color:blue">Analysis of Categorical Variables</span>
# * <span style="color:blue">Analysis of Numerical Variables</span>
# * <span style="color:blue">Analysis of Target Variable</span>
# * <span style="color:blue">Analysis of Correlation</span>
# 
# <h2 style='background:#11489c; border:0; color:white'><center>In particular, EDA consists of</center></h2>
# 
# Organizing and summarizing the raw data, discovering important features and patterns in the data and any striking deviations from those patterns, and then interpreting our findings in the context of the problem
# 
# And can be useful for:
# 
# * Describing the distribution of a single variable (center, spread, shape, outliers)
# * Checking data (for errors or other problems)
# * Checking assumptions to more complex statistical analyses
# * Investigating relationships between variables
# 
# <h2 style='background:#11489c; border:0; color:white'><center>Importing Libraries</center></h2>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.pandas.set_option('display.max_columns', None)


# In[ ]:


df = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.index


# In[ ]:


df.describe().T


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.isnull().sum()


# In[ ]:


def check_df(dataframe, head=5):
    print("##### SHAPE #####")
    print(dataframe.shape)
    print("##### TYPES #####")
    print(dataframe.dtypes)
    print("##### HEAD ######")
    print(dataframe.head(head))
    print("##### TAIL #####")
    print(dataframe.tail(head))
    print("##### NA #####")
    print(dataframe.isnull().sum())
    print("##### QUANTILES #####")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# In[ ]:


check_df(df)


# In[ ]:


df["Sex"].value_counts()


# In[ ]:


df["Sex"].unique()


# In[ ]:


df["Sex"].nunique()


# In[ ]:


cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtypes == "O"]
cat_cols = [col for col in cat_cols if col not in cat_but_car]


# In[ ]:


num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes != "O"]


# In[ ]:


cat_cols = cat_cols + num_but_cat


# In[ ]:


cat_cols


# In[ ]:


df[cat_cols]


# In[ ]:


df[cat_cols].nunique()


# In[ ]:


df[cat_but_car].nunique()


# In[ ]:


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# In[ ]:


for col in cat_cols:
    cat_summary(df, col, plot=True)


# In[ ]:


# analysis of numerical variables
df[["Age", "Fare"]].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T


# In[ ]:


num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in ["PassengerId"]]
num_cols = [col for col in num_cols if col not in cat_cols]


# In[ ]:


num_cols


# In[ ]:


def num_summary(dataframe, numerical_col, plot=False, plot_type="hist"):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        if plot_type == "hist":
            dataframe[numerical_col].hist(bins=30)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()
        
        elif plot_type == "box_plot":
            sns.boxplot(x=dataframe[numerical_col])
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()
        else:
            print("Not correct chart type")


# In[ ]:


num_summary(df, "Age", plot=True)


# In[ ]:


for col in num_cols:
    num_summary(df, col, plot=True)


# In[ ]:


for col in num_cols:
    num_summary(df, col, plot=True, plot_type="box_plot")


# In[ ]:


def grab_col_name(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.
    
    Parameters
    ----------
    dataframe: dataframe
        The dataframe from which variable names are to be retrieved
    cat_th: int, optional
        Class threshold value for numeric but categorical variables
    car_th: int, optional
    
    Returns
    -------
        cat_cols: list
   Categorical variable list
         num_cols: list
             Numeric variable list
         cat_but_car: list
             Categorical view cardinal variable list
    
     Examples
     --------
         import seaborn as sns
         df = sns.load_dataset("iris")
         print(grab_col_names(df))
    
     Notes
     -----
         cat_cols + num_cols + cat_but_Car = total number of variables
         num_but_cat is inside cat_cols.
         The sum of the 3 returned lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables
        
    """
    
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and 
                   dataframe[col].dtypes != "O"]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and 
                   dataframe[col].dtypes == "O"]
    
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    
    return cat_cols, num_cols, cat_but_car


# In[ ]:


grab_col_name(df)


# In[ ]:


dff = pd.read_csv("../input/nba-players-data/all_seasons.csv")


# In[ ]:


cat_cols, num_cols, cat_but_car = grab_col_name(dff)


# In[ ]:


for col in cat_cols:
    cat_summary(dff, col, plot=True)


# In[ ]:


for col in num_cols:
    num_summary(dff, col, plot=True)


# * <span style="color:blue">Analysis of target variable</span>

# In[ ]:


cat_cols, num_cols, cat_but_car = grab_col_name(df)


# In[ ]:


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


# In[ ]:


for col in cat_cols:
    target_summary_with_cat(df, "Survived", col)


# - <span style="color:blue">Analysis of target variable with numerical variables</span>

# In[ ]:


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# In[ ]:


for col in num_cols:
    target_summary_with_num(df, "Survived", col)


# In[ ]:


df_bc = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


df_bc.head()


# In[ ]:


df_bc = df_bc.iloc[:, 1:-1]


# In[ ]:


num_cols = [col for col in df_bc.columns if df_bc[col].dtype in [int, float]]


# In[ ]:


corr = df_bc[num_cols].corr()


# In[ ]:


corr


# In[ ]:


sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


# In[ ]:


# deletion of highly correlated variables
pd.set_option('display.max_columns', 5)

cor_matrix = df_bc.corr().abs()


# In[ ]:


cor_matrix


# In[ ]:


upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))


# In[ ]:


upper_triangle_matrix


# In[ ]:


drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]


# In[ ]:


cor_matrix[drop_list]


# In[ ]:


df_bc.drop(drop_list, axis=1)


# In[ ]:


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


# In[ ]:


high_correlated_cols(df_bc, plot=True)


# In[ ]:


drop_list = high_correlated_cols(df_bc)


# In[ ]:


df_bc.drop(drop_list, axis=1)


# In[ ]:


high_correlated_cols(df_bc.drop(drop_list, axis=1), plot=True)


# In[ ]:


df_fraud = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")


# In[ ]:


df_fraud.head()


# In[ ]:


check_df(df_fraud)


# In[ ]:


drop_list = high_correlated_cols(df_fraud, plot=True)


# In[ ]:


len(df_fraud.drop(drop_list, axis=1).columns)

