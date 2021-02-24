# This is a study about feature engineering in hitters dataset

# Hitters dataset
# Major League Baseball Data from the 1986 and 1987 seasons.

import pandas as pd
import numpy as np
from helpers.data_prep import *
from helpers.eda import *
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)



# Read data

df_ = pd.read_csv(r"D:\DATA SCIENCE\VAHİT BAŞKAN\7. Hafta\odev_1_feature_engineering\hitters.csv")
df = df_.copy()


# Usual check
df.shape
df.head()
df.tail()
df.info()

# Lowercase all to all column names

df.columns = [col.upper() for col in df.columns]

# NAN VALUES

missing_values_table(df, na_name=False)

# Dropping directly NA values because only target variable has NA values
df.dropna(inplace=True)
df.shape

# OUTLIERS

df.describe().T

# Getting categorical columns and numerical columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

# There are outlier observations in this dataset.

#for numerical_column in num_cols:
#    df.boxplot(column=numerical_column)
#    plt.show()

# Getting outlier observations

for numerical_column in num_cols:
    if check_outlier(df, numerical_column,0.05,0.95):
        grab_outliers(df, numerical_column,0.05,0.95)

# I will change the outliers values to %95 value to keep the effect

for numerical_column in num_cols:
    if check_outlier(df, numerical_column):
        replace_with_thresholds(df,numerical_column,0.05,0.95)

check_df(df)

for numerical_column in num_cols:
    df.hist(numerical_column)
    plt.show()


for cat_col in cat_cols:
    cat_summary(df,cat_col)

# Burası ileride editlenecek.
# Çarpik dağılımlar için döünştüme işlemleri uygulanabilir. Log trafsorm vb.


# Creating new variables

df.columns
df["NEW_HITS/ATBAT"] = df["HITS"] / df["ATBAT"]
df["NEW_HMRUN/ATBAT"] = df["HMRUN"] / df["ATBAT"]
df["NEW_HMRUN/HITS"] = df["HMRUN"] / df["HITS"]
df["NEW_RUNS/HITS"] = df["RUNS"] / df["HITS"]
df["NEW_RUNS/ATBAT"] = df["RUNS"] / df["ATBAT"]
df["NEW_WALKS/ATBAT"] = df["WALKS"] / df["ATBAT"]
df["NEW_RBI/ATBAT"] = df["RBI"] / df["ATBAT"]
df["NEW_RBI/HITS"] = df["RBI"] / df["HITS"]
#df["NEW_ERRORS/HMRUN"] = df["ERRORS"] / df["HMRUN"]
#df.drop(columns=["NEW_ERRORS/HMRUN","NEW_HMRUN/ERRORS"],inplace=True)

df["NEW_ATBAT/CATBAT"] = df["ATBAT"] / df["CATBAT"]
df["NEW_HITS/CHITS"] = df["HITS"] / df["CHITS"]
df["NEW_HMRUN/CHMRUN"] = df["HMRUN"] / df["CHMRUN"]
df["NEW_HMRUN/CHMRUN"] = df["HMRUN"] / df["CHMRUN"]
df["NEW_RBI/CRBI"] = df["RBI"] / df["CRBI"]
df["NEW_WALKS/CWALKS"] = df["WALKS"] / df["CWALKS"]

df.loc[(df['YEARS'] > 0) & (df['YEARS'] < 5), 'NEW_EXPERIANCE'] = 1
df.loc[(df['YEARS'] >= 5) & (df['YEARS'] < 10), 'NEW_EXPERIANCE'] = 2
df.loc[(df['YEARS'] >= 10) & (df['YEARS'] < 15), 'NEW_EXPERIANCE'] = 3
df.loc[(df['YEARS'] >= 15) & (df['YEARS'] < 20), 'NEW_EXPERIANCE'] = 4
df.loc[df['YEARS'] >= 20 , 'NEW_EXPERIANCE'] = 5

check_df(df)

rare_analyser(df,"SALARY",0.05)

df = one_hot_encoder(df,cat_cols,drop_first=True)

check_df(df)







