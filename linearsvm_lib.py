import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X, y = X[y != 2, :2], y[y != 2]

X = StandardScaler().fit_transform(X)

m = SVC(kernel='linear', C=1.0)
m.fit(X, y)

acc = accuracy_score(y, m.predict(X))
w = m.coef_[0]
margin = 1 / np.sqrt(np.sum(w**2))

print(f"Linear SVM (Sklearn) Accuracy: {acc:.4f}")
print(f"Calculated Margin: {margin:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7, edgecolors='k')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = m.decision_function(xy).reshape(XX.shape)

ax.contour(
    XX,
    YY,
    Z,
    colors='k',
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=['--', '-', '--'],
)

ax.scatter(
    m.support_vectors_[:, 0],
    m.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors='none',
    edgecolors='k',
    label='Support Vectors',
)

plt.title('Linear SVM (Sklearn)')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.savefig('q11_sklearn_svm.png', bbox_inches='tight', dpi=300)
plt.show()
