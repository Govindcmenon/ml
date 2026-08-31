import numpy as np
import pandas as pd

df = pd.read_csv('Online_Retail.csv', encoding='unicode_escape').dropna(
    subset=['CustomerID']
)
df['TotalSum'] = df['Quantity'] * df['UnitPrice']
rfm = df.groupby('CustomerID').agg(
    {'Quantity': 'sum', 'TotalSum': 'sum', 'InvoiceNo': 'count'}
)
rfm.columns = ['Quantity', 'Monetary', 'Frequency']

X = rfm[['Quantity', 'Frequency']].values
y = (rfm['Monetary'] > rfm['Monetary'].median()).astype(int).values

np.random.seed(42)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def entropy(y):
  counts = np.bincount(y)
  p = counts[counts > 0] / len(y)
  return -np.sum(p * np.log2(p))

class Node:

  def __init__(
      self,
      feature=None,
      threshold=None,
      left=None,
      right=None,
      value=None,
  ):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value


class DT:

  def __init__(self, max_depth=5):
    self.max_depth = max_depth

  def fit(self, X, y, depth=0):
    if len(y) == 0:
      return Node(value=0)
    if len(np.unique(y)) == 1 or depth == self.max_depth:
      return Node(value=np.bincount(y).argmax())

    best_gain = -1
    best_split = None
    current_entropy = entropy(y)

    for f in range(X.shape[1]):
      thresholds = np.unique(X[:, f])
      for t in thresholds:
        left_idx = X[:, f] <= t
        right_idx = X[:, f] > t

        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
          continue

        n = len(y)
        n_l, n_r = len(y[left_idx]), len(y[right_idx])
        e_l, e_r = entropy(y[left_idx]), entropy(y[right_idx])

        # Information Gain (ID3)
        ig = current_entropy - (n_l / n * e_l + n_r / n * e_r)

        if ig > best_gain:
          best_gain = ig
          best_split = (f, t, left_idx, right_idx)

    if best_gain > 0 and best_split is not None:
      f, t, left_idx, right_idx = best_split
      left = self.fit(X[left_idx], y[left_idx], depth + 1)
      right = self.fit(X[right_idx], y[right_idx], depth + 1)
      return Node(feature=f, threshold=t, left=left, right=right)

    return Node(value=np.bincount(y).argmax())

  def predict(self, node, X):
    preds = []
    for x in X:
      curr = node
      while curr.value is None:
        if x[curr.feature] <= curr.threshold:
          curr = curr.left
        else:
          curr = curr.right
      preds.append(curr.value)
    return np.array(preds)

model = DT(max_depth=5)
tree = model.fit(X_train, y_train)

p = model.predict(tree, X_test)
acc = np.mean(p == y_test)
print(f"Decision Tree (Scratch - ID3) Accuracy: {acc:.4f}")
