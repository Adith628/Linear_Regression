from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import seaborn as sns

companies = pd.read_csv('1000_Companies.csv')
x = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values

companies.head()

# Encoding categorical data
labelencoder = LabelEncoder()
x[:, 3] =labelencoder.fit_transform(x[:, 3])
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)

# Removing dummy variable trap
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(x_test)

# calculating the coefficients
print(regressor.coef_)
# calculating the intercept
print(regressor.intercept_)

# calculating the R squared value
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(score)