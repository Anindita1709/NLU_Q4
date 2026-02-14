from data_loader import load_dataset
from preprocess import clean_text
from features import build_vocabulary, bow_matrix, tfidf_matrix

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import os

print("Loading dataset...")
df = load_dataset()

print("Preprocessing text...")
df["clean"] = df["text"].apply(clean_text)

docs = df["clean"].tolist()   # list of tokenized docs
labels = df["label"].values

train_idx, test_idx = train_test_split(
    np.arange(len(docs)), test_size=0.2, random_state=42, stratify=labels
)

y_train = labels[train_idx]
y_test  = labels[test_idx]

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": LinearSVC()
}

results = []

print("\nRunning Bag of Words...")
vocab, word2idx = build_vocabulary(docs, ngram_range=(1,1))
X = bow_matrix(docs, word2idx, ngram_range=(1,1))

X_train = X[train_idx]
X_test  = X[test_idx]

for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append(["Bag of Words", model_name, acc])
    print(model_name, ":", acc)

print("\nRunning N-grams...")
vocab, word2idx = build_vocabulary(docs, ngram_range=(1,2))
X = bow_matrix(docs, word2idx, ngram_range=(1,2))

X_train = X[train_idx]
X_test  = X[test_idx]

for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append(["N-grams", model_name, acc])
    print(model_name, ":", acc)

print("\nRunning TF-IDF...")
vocab, word2idx = build_vocabulary(docs, ngram_range=(1,2))
X_bow = bow_matrix(docs, word2idx, ngram_range=(1,2))
X = tfidf_matrix(X_bow)

X_train = X[train_idx]
X_test  = X[test_idx]

for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append(["TF-IDF", model_name, acc])
    print(model_name, ":", acc)

#result
results_df = pd.DataFrame(results, columns=["Feature","Model","Accuracy"])

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nFinal Results:")
print(results_df)
print("\nSaved to results/model_comparison.csv")
