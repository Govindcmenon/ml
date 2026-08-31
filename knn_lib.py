import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

accs = []
ks = list(range(1, 15))
for k in ks:
  m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
  acc = accuracy_score(y_test, m.predict(X_test))
  accs.append(acc)

best_k = ks[np.argmax(accs)]
print(f"Optimal K: {best_k} with Accuracy: {max(accs):.4f}\n")
for k, acc in zip(ks, accs):
  print(f"K = {k:2d}, Accuracy = {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(ks, accs, marker='o', linestyle='-', color='b')
plt.title('KNN Accuracy vs K (sklearn)')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('q8_sklearn_knn_plot.png')
plt.show()
