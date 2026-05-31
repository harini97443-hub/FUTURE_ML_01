# Sales Forecasting Project
# Task 1: Forecast future sales using historical business data

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample sales data
data = {
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [200, 250, 300, 350, 400, 450]
}

df = pd.DataFrame(data)

# Prepare data
X = df[['Month']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
future_months = pd.DataFrame({'Month': [7, 8, 9]})
predictions = model.predict(future_months)

# Print predictions
print("Future Sales Forecast:")
for month, pred in zip(future_months['Month'], predictions):
    print(f"Month {month}: {pred:.2f}")

# Plot graph
plt.plot(df['Month'], df['Sales'], label='Historical Sales')
plt.plot(future_months['Month'], predictions, label='Forecast')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.show()
