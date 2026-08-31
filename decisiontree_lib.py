import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('Online_Retail.csv', encoding='unicode_escape').dropna(
    subset=['CustomerID']
)
df['TotalSum'] = df['Quantity'] * df['UnitPrice']
rfm = df.groupby('CustomerID').agg(
    {'Quantity': 'sum', 'TotalSum': 'sum', 'InvoiceNo': 'count'}
)
rfm.columns = ['Quantity', 'Monetary', 'Frequency']

X = rfm[['Quantity', 'Frequency']]
y = (rfm['Monetary'] > rfm['Monetary'].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

m = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
m.fit(X_train, y_train)

acc = m.score(X_test, y_test)
print(f"Decision Tree (Sklearn) Accuracy: {acc:.4f}")
print("Feature Importances:", m.feature_importances_)

plt.figure(figsize=(15, 10))
plot_tree(
    m,
    filled=True,
    feature_names=X.columns,
    class_names=['Low', 'High'],
    fontsize=8,
)
plt.title("Decision Tree Visualization (max_depth=5)", fontsize=14)
plt.savefig('q9_sklearn_tree.png', bbox_inches='tight', dpi=300)
plt.show()
