import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
np.random.seed(42)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

mu, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

def knn(X_tr, y_tr, X_te, k):
  preds = []
  for x in X_te:
    dists = np.sqrt(np.sum((X_tr - x) ** 2, axis=1))

    nearest = y_tr[np.argsort(dists)[:k]]

    vals, counts = np.unique(nearest, return_counts=True)
    preds.append(vals[np.argmax(counts)])

  return np.array(preds)

accs = []
ks = list(range(1, 15))
for k in ks:
  p = knn(X_train, y_train, X_test, k)
  accs.append(np.mean(p == y_test))

best_k = ks[np.argmax(accs)]
print(f"Optimal K: {best_k} with Accuracy: {max(accs):.4f}\n")
for k, acc in zip(ks, accs):
  print(f"K = {k:2d}, Accuracy = {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(ks, accs, marker='o', linestyle='-', color='b')
plt.title('KNN Accuracy vs K')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('q8_scratch_knn_plot.png')
plt.show()
