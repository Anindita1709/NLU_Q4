from data_loader import load_dataset
from preprocess import clean_text
from features import build_vocabulary, bow_matrix, tfidf_matrix

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df = load_dataset()
df["clean"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": LinearSVC()
}

features = {
    "Bag of Words": bow_features(),
    "N-grams": ngram_features(),
    "TF-IDF": tfidf_features()
}

results = []

for feature_name, vectorizer in features.items():

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)

        results.append([feature_name, model_name, acc])

results_df = pd.DataFrame(results, columns=["Feature","Model","Accuracy"])
print(results_df)

results_df.to_csv("results/model_comparison.csv", index=False)
