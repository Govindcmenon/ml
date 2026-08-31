import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X, y = X[y != 2, :2], y[y != 2]
y = np.where(y == 0, -1, 1)

mu, std = X.mean(axis=0), X.std(axis=0)
X = (X - mu) / std
w = np.zeros(2)
b = 0.0
lr = 0.01
epochs = 1000
C = 1.0

for _ in range(epochs):
  for i, x_i in enumerate(X):
    # Functional margin condition: y_i * (w · x_i + b) >= 1
    condition = y[i] * (np.dot(x_i, w) + b) >= 1
    if condition
      w -= lr * (w / epochs)
    else:
      w -= lr * (w / epochs - C * y[i] * x_i)
      b -= lr * (-C * y[i])

preds = np.where(np.dot(X, w) + b >= 0, 1, -1)
acc = np.mean(preds == y)
margin = 1 / np.sqrt(np.sum(w**2) + 1e-8)

print(f"Linear SVM (Scratch) Accuracy: {acc:.4f}")
print(f"Calculated Margin: {margin:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7, edgecolors='k')

ax = plt.gca()
xlim = ax.get_xlim()
xx = np.linspace(xlim[0], xlim[1], 30)

# Decision Boundary: w0*x0 + w1*x1 + b = 0  =>  x1 = (-b - w0*x0) / w1
yy = (-b - w[0] * xx) / (w[1] + 1e-8)

# Margin lines: w0*x0 + w1*x1 + b = ±1  =>  x1 = (±1 - b - w0*x0) / w1
yy_down = (-1 - b - w[0] * xx) / (w[1] + 1e-8)
yy_up = (1 - b - w[0] * xx) / (w[1] + 1e-8)

plt.plot(xx, yy, 'k-', label='Decision Boundary')
plt.plot(xx, yy_down, 'k--', label='Margin Boundary (-1)')
plt.plot(xx, yy_up, 'k--', label='Margin Boundary (+1)')

plt.title('Linear SVM (Scratch) on Iris Dataset')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.grid(True)
plt.savefig('q11_scratch_svm.png', bbox_inches='tight', dpi=300)
plt.show()
