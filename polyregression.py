import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
cols = [
    'mpg',
    'cylinders',
    'displacement',
    'horsepower',
    'weight',
    'acceleration',
    'model_year',
    'origin',
]
df = (
    pd.read_csv(url, sep=r'\s+', names=cols)
    .replace('?', np.nan)
    .dropna()
    .astype({'horsepower': float})
)

X_raw = df['displacement'].values
y = df['mpg'].values

split = int(0.8 * len(X_raw))
X_tr_raw, X_te_raw = X_raw[:split], X_raw[split:]
y_tr, y_te = y[:split], y[split:]

mu, std = X_tr_raw.mean(), X_tr_raw.std()
X_tr, X_te = (X_tr_raw - mu) / std, (X_te_raw - mu) / std

def fit_poly(X_train, y_train, degree):
  X_poly = np.column_stack([X_train**deg for deg in range(degree + 1)])
  theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y_train
  return theta

def predict_poly(X_test, theta):
  X_poly = np.column_stack([X_test**deg for deg in range(len(theta))])
  return X_poly @ theta

def metrics(y_true, y_pred):
  mse = np.mean((y_true - y_pred) ** 2)
  r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
  return mse, r2

degrees = [1, 2, 3]
plt.figure(figsize=(8, 5))
plt.scatter(X_te_raw, y_te, alpha=0.4, label='Test Data', color='gray')

sort_idx = np.argsort(X_te_raw)
X_line = np.linspace(X_te_raw.min(), X_te_raw.max(), 200)
X_line_s = (X_line - mu) / std

for d in degrees:
  theta = fit_poly(X_tr, y_tr, degree=d)
  y_pred = predict_poly(X_te, theta)
  mse, r2 = metrics(y_te, y_pred)

  label_name = "Linear (Deg 1)" if d == 1 else f"Poly Deg {d}"
  print(f"{label_name:15s} - MSE: {mse:.4f}, R2: {r2:.4f}")

  y_line = predict_poly(X_line_s, theta)
  plt.plot(X_line, y_line, lw=2, label=f"{label_name} (R2={r2:.2f})")

plt.title('Auto MPG: Polynomial vs Linear Regression (Scratch)')
plt.xlabel('Displacement')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True)
plt.show()
