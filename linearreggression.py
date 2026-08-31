import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
mask = X[:, 2] < 15  # Remove extreme outliers in AveRooms
X, y = X[mask, 2:3], y[mask]

split = int(0.8 * len(X))
X_tr_raw, X_te_raw = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

mu, std = X_tr_raw.mean(), X_tr_raw.std()
X_tr, X_te = (X_tr_raw - mu) / std, (X_te_raw - mu) / std

X_b_tr, X_b_te = np.c_[np.ones(len(X_tr)), X_tr], np.c_[np.ones(len(X_te)), X_te]

def metrics(y_true, y_pred):
  mse = np.mean((y_true - y_pred) ** 2)
  r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
  return mse, r2


#Method A: Gradient Descent
theta_gd = np.zeros(2)
lr, epochs = 0.01, 1000
for _ in range(epochs):
  theta_gd -= (lr / len(y_tr)) * (X_b_tr.T @ (X_b_tr @ theta_gd - y_tr))

#Method B: Normal Equation -> theta = (X^T * X)^(-1) * X^T * y
theta_norm = np.linalg.inv(X_b_tr.T @ X_b_tr) @ X_b_tr.T @ y_tr

mse_gd, r2_gd = metrics(y_te, X_b_te @ theta_gd)
mse_norm, r2_norm = metrics(y_te, X_b_te @ theta_norm)

print(f"Gradient Descent - MSE: {mse_gd:.4f}, R2: {r2_gd:.4f}")
print(f"Normal Equation  - MSE: {mse_norm:.4f}, R2: {r2_norm:.4f}")

plt.scatter(X_te_raw, y_te, alpha=0.3, label='Data')
plt.plot(
    X_te_raw[np.argsort(X_te_raw[:, 0])],
    (X_b_te @ theta_gd)[np.argsort(X_te_raw[:, 0])],
    'r-',
    lw=2,
    label='Fitted Line',
)
plt.xlabel('AveRooms')
plt.ylabel('MedHouseVal')
plt.legend()
plt.show()
