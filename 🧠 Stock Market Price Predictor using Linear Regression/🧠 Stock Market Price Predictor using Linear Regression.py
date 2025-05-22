import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Download stock data
ticker = "AAPL"  # You can change this to any stock symbol
df = yf.download(ticker, start="2015-01-01", end="2024-12-31")
df = df[['Close']]

# 2. Create features (shifted close price to predict next day)
df['Prediction'] = df[['Close']].shift(-1)

# 3. Prepare the data
X = np.array(df.drop(['Prediction'], axis=1))[:-1]  # All rows except last
y = np.array(df['Prediction'])[:-1]  # All rows except last

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Test the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# 7. Predict the next day
real_data = np.array(df.drop(['Prediction'], axis=1))[-1:].reshape(1, -1)
next_day_prediction = model.predict(real_data)
print(f"Predicted next day's closing price: ${next_day_prediction[0]:.2f}")

# 8. Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], label='Actual Price', marker='o')
plt.plot(predictions[:50], label='Predicted Price', marker='x')
plt.title(f"{ticker} Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()