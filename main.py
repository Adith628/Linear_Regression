from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

companies = pd.read_csv('1000_Companies.csv')
x = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values

companies.head()

sns.heatmap(companies.corr())

# Encoding categorical data
labelencoder = LabelEncoder()
x[:, 3] =labelencoder.fit_transform(x[:, 3])
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)
