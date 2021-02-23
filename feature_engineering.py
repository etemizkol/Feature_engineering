
import numpy as pd
import math
import statsmodels.stats.api as sms
import pandas as pd

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 999)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)

# titanic, hitters,diabetes verisetleri üzerinde feature engineering çalışmalarını yürütünüz

# Feature Engineering - Data Pre-Processing
# kem küm


# ------------------------------------------------Titanic-----------------------------------------------------------#


df_titanic_ = pd.read_csv(r"D:\DATA SCIENCE\VAHİT BAŞKAN\7. Hafta\odev_1_feature_engineering\titanic.csv")
df_titanic = df_titanic_.copy()

# Değişken Mühendisliği

# Kabini NA olanlar için CABIN BOOL
df_titanic["NEW_CABIN_BOOL"] = df_titanic["Cabin"].isnull().astype("int")
# Name Letter Count
df_titanic["NEW_NAME_COUNT"] = df_titanic["Name"].str.len()
# Name word Count
df_titanic["NEW_NAME_WORD_COUNT"] = df_titanic["Name"].apply(lambda x: len(str(x).split(" ")))
# isDoctor ?
df_titanic["NEW_NAME_DR"] = df_titanic["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# Name Titles
df_titanic["NEW_TITLE"] = df_titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# Familiy Size
df_titanic["NEW_FAMILY_SIZE"] = df_titanic["SibSp"] + df_titanic["Parch"] + 1
# AGE * PLCASS
df_titanic["NEW_AGE_PCLASS"] = df_titanic["Age"] * df_titanic["Pclass"]
# isAlone ?
df_titanic.loc[((df_titanic['SibSp'] + df_titanic['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df_titanic.loc[((df_titanic['SibSp'] + df_titanic['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
# Age to categorical
df_titanic.loc[(df_titanic['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df_titanic.loc[(df_titanic['Age'] >= 18) & (df_titanic['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df_titanic.loc[(df_titanic['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# Age Categorical * Sex
df_titanic.loc[(df_titanic['Sex'] == 'male') & (df_titanic['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df_titanic.loc[(df_titanic['Sex'] == 'male') & ((df_titanic['Age'] > 21) & (df_titanic['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df_titanic.loc[(df_titanic['Sex'] == 'male') & (df_titanic['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df_titanic.loc[(df_titanic['Sex'] == 'female') & (df_titanic['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df_titanic.loc[(df_titanic['Sex'] == 'female') & ((df_titanic['Age'] > 21) & (df_titanic['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df_titanic.loc[(df_titanic['Sex'] == 'female') & (df_titanic['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df_titanic.head()
df_titanic.shape

# convert uppercase to all columns

df_titanic.columns = [col.upper() for col in df_titanic.columns]

# Outliers

num_cols = [col for col in df_titanic.columns if len(df_titanic[col].unique()) > 20
            and df_titanic[col].dtypes != 'O'
            and col not in "PASSENGERID"]

from helpers.data_prep import check_outlier

for col in num_cols:
    print(col, check_outlier(df_titanic, col))

from helpers.data_prep import replace_with_thresholds

for col in num_cols:
    replace_with_thresholds(df_titanic, col)
#replace_with_thresholds(df_titanic, "FARE")

df_titanic.describe().T

for col in num_cols:
    print(col, check_outlier(df_titanic, col))

from helpers.eda import check_df
check_df(df_titanic)

# Missing Values

from helpers.data_prep import missing_values_table
missing_values_table(df_titanic)

df_titanic.drop("CABIN", inplace=True, axis=1)
remove_vars = ["TICKET", "NAME"]
df_titanic.drop(remove_vars, inplace=True, axis=1)

# Filling age values from grouped by NEW_TITLE
missing_values_table(df_titanic)
df_titanic["AGE"] = df_titanic["AGE"].fillna(df_titanic.groupby("NEW_TITLE")["AGE"].transform("median"))

missing_values_table(df_titanic)

# Re-assign NEW_AGE_CAT and NEW_AGE_PCLASS

df_titanic["NEW_AGE_PCLASS"] = df_titanic["AGE"] * df_titanic["PCLASS"]
df_titanic.loc[(df_titanic['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df_titanic.loc[(df_titanic['AGE'] >= 18) & (df_titanic['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df_titanic.loc[(df_titanic['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df_titanic.loc[(df_titanic['SEX'] == 'male') & (df_titanic['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df_titanic.loc[(df_titanic['SEX'] == 'male') & ((df_titanic['AGE'] > 21) & (df_titanic['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df_titanic.loc[(df_titanic['SEX'] == 'male') & (df_titanic['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df_titanic.loc[(df_titanic['SEX'] == 'female') & (df_titanic['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df_titanic.loc[(df_titanic['SEX'] == 'female') & ((df_titanic['AGE'] > 21) & (df_titanic['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df_titanic.loc[(df_titanic['SEX'] == 'female') & (df_titanic['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# To fill EMBARKED with Mode
df_titanic = df_titanic.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

# Label Encoding

binary_cols = [col for col in df_titanic.columns if len(df_titanic[col].unique()) == 2 and df_titanic[col].dtypes == 'O']

from helpers.data_prep import label_encoder

for col in binary_cols:
    df_titanic = label_encoder(df_titanic, col)

# Rare Encoding

from helpers.data_prep import rare_analyser

rare_analyser(df_titanic, "SURVIVED", 0.01)

from helpers.data_prep import rare_encoder

df_titanic = rare_encoder(df_titanic, 0.01)

rare_analyser(df_titanic, "SURVIVED", 0.01)

df_titanic["NEW_TITLE"].value_counts()

# One Hot Encoding

ohe_cols = [col for col in df_titanic.columns if 10 >= len(df_titanic[col].unique()) > 2]

from helpers.data_prep import one_hot_encoder
df_titanic = one_hot_encoder(df_titanic, ohe_cols)

df_titanic.head()
df_titanic.shape

# Standart Scaler

check_df(df_titanic)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df_titanic[["AGE"]])
df_titanic["AGE"] = scaler.transform(df_titanic[["AGE"]])

# Model

y = df_titanic["SURVIVED"]
X = df_titanic.drop(["PASSENGERID", "SURVIVED"], axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier().fit(X, y)
y_pred = rf_model.predict(X)
accuracy_score(y_pred, y)

# %99
import seaborn as sns
import matplotlib.pyplot as plt

def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('importances-01.png')
    plt.show()

plot_importance(rf_model, X)