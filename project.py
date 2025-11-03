# Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = X["alcohol"]  # Regression target (continuous)
X = X.drop("alcohol", axis=1)  # Drop target from features

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Random Forest Regression Model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

print("\nRandom Forest Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# Visualization - Actual vs Predicted
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_rf, color='green')
plt.title("Random Forest Regression: Actual vs Predicted")
plt.xlabel("Actual Alcohol %")
plt.ylabel("Predicted Alcohol %")
plt.grid(True)
plt.show()

# Error Distribution
errors = y_test - y_pred_rf
plt.figure(figsize=(6,5))
sns.histplot(errors, kde=True, color='purple')
plt.title("Regression Error Distribution")
plt.xlabel("Prediction Error")
plt.show()
