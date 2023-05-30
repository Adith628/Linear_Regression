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

# Removing dummy variable trap
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import reain_test_spilt 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to the training set

