import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df[['AveRooms']].values
y = df['MedHouseVal'].values

valid_idx = X.flatten() < 15
X, y = X[valid_idx], y[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Scikit-Learn Linear Regression Metrics:")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"R-squared Score (R2)     : {r2:.4f}")
print(f"Intercept                : {model.intercept_:.4f}")
print(f"Slope (Coefficient)      : {model.coef_[0]:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, alpha=0.3, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
plt.title('Scikit-Learn Linear Regression: AveRooms vs Housing Price')
plt.xlabel('Average Rooms per Dwelling (AveRooms)')
plt.ylabel('Median House Value ($100,000s)')
plt.legend()
plt.grid(True)
plt.savefig('california_lr_sklearn.png', bbox_inches='tight', dpi=300)
plt.show()
