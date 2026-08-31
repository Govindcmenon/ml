from collections import Counter
import numpy as np
from sklearn.datasets import fetch_20newsgroups

cats = ['alt.atheism', 'comp.graphics', 'sci.space', 'talk.religion.misc']
train = fetch_20newsgroups(
    subset='train', categories=cats, remove=('headers', 'footers', 'quotes')
)
test = fetch_20newsgroups(
    subset='test', categories=cats, remove=('headers', 'footers', 'quotes')
)

words = [w for doc in train.data for w in set(doc.lower().split())]
vocab = {
    w: i for i, (w, _) in enumerate(Counter(words).most_common(5000))
}

def vectorize(docs):
  X = np.zeros((len(docs), len(vocab)), dtype=int)
  for i, doc in enumerate(docs):
    for w in doc.lower().split():
      if w in vocab:
        X[i, vocab[w]] += 1
  return X

X_tr, y_tr = vectorize(train.data), train.target
X_te, y_te = vectorize(test.data), test.target
classes = np.unique(y_tr)

def run_mnb(alpha=1.0):
  prior = np.log([np.mean(y_tr == c) for c in classes])
  log_prob = np.array([
      np.log(
          (X_tr[y_tr == c].sum(axis=0) + alpha)
          / (X_tr[y_tr == c].sum() + alpha * X_tr.shape[1])
      )
      for c in classes
  ])
  preds = classes[np.argmax(X_te @ log_prob.T + prior, axis=1)]
  return np.mean(preds == y_te)

def run_bnb(alpha=1.0):
  prior = np.log([np.mean(y_tr == c) for c in classes])
  X_bin_tr, X_bin_te = (X_tr > 0).astype(int), (X_te > 0).astype(int)
  p = np.array([
      (X_bin_tr[y_tr == c].sum(axis=0) + alpha)
      / (np.sum(y_tr == c) + 2 * alpha)
      for c in classes
  ])
  jll = (
      prior
      + X_bin_te @ np.log(p).T
      + (1 - X_bin_te) @ np.log(1 - p).T
  )
  preds = classes[np.argmax(jll, axis=1)]
  return np.mean(preds == y_te)

print(f"Multinomial NB Accuracy: {run_mnb():.4f}")
print(f"Bernoulli NB Accuracy:   {run_bnb():.4f}")
