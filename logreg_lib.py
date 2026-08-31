import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def eval_model(X_tr, X_te, prefix):
  m = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
  p = m.predict(X_te)
  acc = accuracy_score(y_test, p)
  prec = precision_score(y_test, p)
  rec = recall_score(y_test, p)
  f1 = f1_score(y_test, p)
  print(
      f"{prefix} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall:"
      f" {rec:.4f}, F1: {f1:.4f}"
  )

eval_model(X_train, X_test, "Unscaled")

s = StandardScaler()
eval_model(s.fit_transform(X_train), s.transform(X_test), "Scaled")
