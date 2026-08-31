import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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

X = df[['displacement']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, alpha=0.4, label='Test Data', color='gray')

X_line = np.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)

degrees = [1, 2, 3]
for d in degrees:
  # Pipeline: Polynomial Features -> Scaling -> Linear Regression
  model = make_pipeline(
      PolynomialFeatures(degree=d), StandardScaler(), LinearRegression()
  )
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  label_name = "Linear (Deg 1)" if d == 1 else f"Poly Deg {d}"
  print(f"{label_name:15s} - MSE: {mse:.4f}, R2: {r2:.4f}")

  y_line = model.predict(X_line)
  plt.plot(X_line, y_line, lw=2, label=f"{label_name} (R2={r2:.2f})")

plt.title('Auto MPG: Polynomial vs Linear Regression (sklearn)')
plt.xlabel('Displacement')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True)
plt.savefig('auto_mpg_poly_sklearn.png', bbox_inches='tight', dpi=300)
plt.show()
