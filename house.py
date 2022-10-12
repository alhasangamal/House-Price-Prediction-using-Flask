# import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv("house_price.csv")

# Choose important column

col = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']

data = data[col]

# Spilt data into train and test

X = data.iloc[:, :4]
y = data.iloc[:, 4:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Build model

lr = LinearRegression()
lr.fit(X_train, y_train)

# Save model

pickle.dump(lr, open('model.pkl', 'wb'))

