# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:34:21 2024

@author: youss
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("Unemployment in India.csv")

data

# Drop rows with missing values
data.dropna(inplace=True)

# Check for missing values
data.isnull().sum()

# Visualize the distribution of unemployment rate
sns.histplot(data.iloc[:,3], kde=True)
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()

# Drop irrelevant columns
data.drop(['Region', ' Frequency','Area'], axis=1, inplace=True)
# Convert 'Date' column to datetime format
data[' Date'] = pd.to_datetime(data[' Date'])

# Set 'Date' column as index
data.set_index(' Date', inplace=True)

# Plotting time series of unemployment rate
sns.histplot(data[' Estimated Unemployment Rate (%)'], kde=True)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()

# Define features and target
X = data.drop(' Estimated Unemployment Rate (%)', axis=1)
y = data[' Estimated Unemployment Rate (%)']
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
# Evaluation metrics
print("Training MSE:", mean_squared_error(y_train, train_preds))
print("Testing MSE:", mean_squared_error(y_test, test_preds))
print("Training R^2:", r2_score(y_train, train_preds))
print("Testing R^2:", r2_score(y_test, test_preds))

