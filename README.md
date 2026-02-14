# NLU_Q4

# Sport vs Politics Text Classifier

This project builds a Machine Learning classifier that reads a news article(raw text file) and classifies it as **Sport** or **Politics**.

## Techniques Used
Feature Representation:
- Bag of Words
- n-grams
- TF-IDF

Machine Learning Models:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

Best Model:
TF-IDF + Linear SVM

---

## Dataset

Dataset was collected from kaggle BBC news articles.

| Class | Files |
|------|------|
| Sport | 045, 051, 079.. |
| Politics | 130, 131, 244.. |

---

## Installation

```bash
pip install -r requirements.txt
