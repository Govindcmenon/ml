import numpy as np
import pandas as pd

df = pd.read_csv('adult.csv').replace('?', np.nan).dropna()
X = df.select_dtypes(include=[np.number]).values
y = (df['income'].str.contains('>50K')).astype(int).values

np.random.seed(42)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

mu, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_s = (X_train - mu) / (std + 1e-8)
X_test_s = (X_test - mu) / (std + 1e-8)

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

w = np.zeros(X_train_s.shape[1])
b = 0
lr = 0.1
iters = 1000

for _ in range(iters):
  z = np.clip(np.dot(X_train_s, w) + b, -250, 250)
  p = 1 / (1 + np.exp(-z))
  w -= lr * np.dot(X_train_s.T, (p - y_train)) / len(y_train)
  b -= lr * np.sum(p - y_train) / len(y_train)

z_test = np.clip(np.dot(X_test_s, w) + b, -250, 250)
p_lr = (1 / (1 + np.exp(-z_test)) >= 0.5).astype(int)
acc_lr, prec_lr, rec_lr, f1_lr = metrics(y_test, p_lr)
print(
    f"LR - Accuracy: {acc_lr:.4f}, Precision: {prec_lr:.4f}, Recall:"
    f" {rec_lr:.4f}, F1: {f1_lr:.4f}"
)
def entropy(y):
  if len(y) == 0:
    return 0
  counts = np.bincount(y)
  p = counts[counts > 0] / len(y)
  return -np.sum(p * np.log2(p))

ce = entropy(y_train)
best_gain = -1
best_split = None

for f in range(X_train.shape[1]):
  thresholds = np.percentile(X_train[:, f], [25, 50, 75])
  for t in thresholds:
    l = y_train[X_train[:, f] <= t]
    r = y_train[X_train[:, f] > t]
    if len(l) == 0 or len(r) == 0:
      continue

    ig = ce - (
        len(l) / len(y_train) * entropy(l) + len(r) / len(y_train) * entropy(r)
    )
    if ig > best_gain:
      best_gain = ig
      l_val = np.bincount(l).argmax() if len(l) > 0 else 0
      r_val = np.bincount(r).argmax() if len(r) > 0 else 0
      best_split = (f, t, l_val, r_val)

if best_split:
  f, t, val_l, val_r = best_split
  p_dt = np.where(X_test[:, f] <= t, val_l, val_r)
  acc_dt, prec_dt, rec_dt, f1_dt = metrics(y_test, p_dt)
  print(
      f"DT - Accuracy: {acc_dt:.4f}, Precision: {prec_dt:.4f}, Recall:"
      f" {rec_dt:.4f}, F1: {f1_dt:.4f}"
  )
