import numpy as np
import pandas as pd

# Load dataset
col_names = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome',
]
df = pd.read_csv('diabetes.csv', header=0, names=col_names)

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

np.random.seed(42)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


class LR:

  def __init__(self, lr=0.01, iters=5000):
    self.lr = lr
    self.iters = iters

  def fit(self, X, y):
    self.w = np.zeros(X.shape[1])
    self.b = 0
    for _ in range(self.iters):
      z = np.clip(np.dot(X, self.w) + self.b, -250, 250)
      p = 1 / (1 + np.exp(-z))
      self.w -= self.lr * np.dot(X.T, (p - y)) / len(y)
      self.b -= self.lr * np.sum(p - y) / len(y)

  def predict(self, X):
    z = np.clip(np.dot(X, self.w) + self.b, -250, 250)
    return (1 / (1 + np.exp(-z)) >= 0.5).astype(int)


def metrics(y_t, y_p):
  tp = np.sum((y_t == 1) & (y_p == 1))
  tn = np.sum((y_t == 0) & (y_p == 0))
  fp = np.sum((y_t == 0) & (y_p == 1))
  fn = np.sum((y_t == 1) & (y_p == 0))

  acc = (tp + tn) / len(y_t)
  prec = tp / (tp + fp) if (tp + fp) > 0 else 0
  rec = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
  return acc, prec, rec, f1

model = LR(lr=0.01, iters=5000)
model.fit(X_train, y_train)
acc, prec, rec, f1 = metrics(y_test, model.predict(X_test))
print(
    f"Unscaled - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall:"
    f" {rec:.4f}, F1: {f1:.4f}"
)

mu, std = X_train.mean(axis=0), X_train.std(axis=0)

std = np.where(std == 0, 1e-8, std)

X_train_s = (X_train - mu) / std
X_test_s = (X_test - mu) / std

model.fit(X_train_s, y_train)
acc, prec, rec, f1 = metrics(y_test, model.predict(X_test_s))
print(
    f"Scaled   - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall:"
    f" {rec:.4f}, F1: {f1:.4f}"
)
