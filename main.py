
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

companies = pd.read_csv('1000_Companies.csv')
x = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values

companies.head()


