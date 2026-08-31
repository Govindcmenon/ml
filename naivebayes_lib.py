from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

categories = [
    'alt.atheism',
    'comp.graphics',
    'sci.space',
    'talk.religion.misc',
]
train_data = fetch_20newsgroups(
    subset='train', categories=categories, remove=('headers', 'footers', 'quotes')
)
test_data = fetch_20newsgroups(
    subset='test', categories=categories, remove=('headers', 'footers', 'quotes')
)

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)
y_train, y_test = train_data.target, test_data.target

# Train and evaluate Multinomial Naïve Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred_mnb = mnb.predict(X_test)
acc_mnb = accuracy_score(y_test, pred_mnb)
f1_mnb = f1_score(y_test, pred_mnb, average='macro')

# Train and evaluate Bernoulli Naïve Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
pred_bnb = bnb.predict(X_test)
acc_bnb = accuracy_score(y_test, pred_bnb)
f1_bnb = f1_score(y_test, pred_bnb, average='macro')

print(
    f"Multinomial NB (sklearn) - Accuracy: {acc_mnb:.4f}, Macro F1:"
    f" {f1_mnb:.4f}"
)
print(
    f"Bernoulli NB   (sklearn) - Accuracy: {acc_bnb:.4f}, Macro F1:"
    f" {f1_bnb:.4f}"
)
