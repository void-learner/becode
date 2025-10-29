import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

print(df.isnull().sum())
print(df.shape)
print(df['label'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

count_vect = CountVectorizer(stop_words='english')
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_tfidf = tfidf_transformer.transform(count_vect.transform(X_test))

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

samples = [
    "Win a free iPhone! Click here to claim your prize now.",
    "Are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Reset your password immediately!",
    "Hey, just checking in — how's your day going?"
]
sample_tfidf = tfidf_transformer.transform(count_vect.transform(samples))
predictions = model.predict(sample_tfidf)

for msg, label in zip(samples, predictions):
    print(f"\nMessage: {msg}\nPredicted: {'SPAM' if label==1 else 'HAM'}")
