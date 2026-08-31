import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('adult.csv').replace('?', np.nan).dropna()
X = df.select_dtypes(include=[np.number])
y = (df['income'].str.contains('>50K')).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

s = StandardScaler()
X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)

lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
dt = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

def eval_m(m, X_te, prefix):
  p = m.predict(X_te)
  acc = accuracy_score(y_test, p)
  prec = precision_score(y_test, p)
  rec = recall_score(y_test, p)
  f1 = f1_score(y_test, p)
  print(
      f"{prefix} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall:"
      f" {rec:.4f}, F1: {f1:.4f}"
  )

eval_m(lr, X_test_s, "LR")
eval_m(dt, X_test, "DT")
