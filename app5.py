# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 23:02:55 2023

@author: amind
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel(r'C:\Users\amind\Downloads\Crude_oil_price-3.xlsx')

# Preprocess the data
dates = pd.to_datetime(data['Date'])
prices = data['Price']

# Train the ARIMA model
model = ARIMA(prices, order=(3, 1, 2))
model_fit = model.fit()

# Make predictions
forecast = model_fit.forecast(steps=30)

# Plot actual vs forecasted prices
plt.figure(figsize=(10, 6))
plt.plot(dates[-30:], prices[-30:], color='blue', label='Actual')
plt.plot(pd.date_range(start=dates.iloc[-1], periods=30, freq='D'), forecast, color='red', linewidth=2, label='Forecast')
plt.title('Actual vs. Forecasted Crude Oil Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot forecasted prices
plt.figure(figsize=(10, 6))
plt.plot(pd.date_range(start=dates.iloc[-1], periods=30, freq='D'), forecast, color='red', linewidth=2, label='Forecast')
plt.title('Forecasted Crude Oil Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

