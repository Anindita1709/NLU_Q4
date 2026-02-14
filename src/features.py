import re
import numpy as np
from collections import Counter, defaultdict

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.split()

def build_vocabulary(docs, ngram_range=(1,1)):
    vocab = set()

    for doc in docs:
        tokens = tokenize(doc)

        for n in range(ngram_range[0], ngram_range[1]+1):
            for i in range(len(tokens)-n+1):
                ngram = " ".join(tokens[i:i+n])
                vocab.add(ngram)

    vocab = sorted(list(vocab))
    word2idx = {word:i for i,word in enumerate(vocab)}

    return vocab, word2idx

def bow_matrix(docs, word2idx, ngram_range=(1,1)):
    matrix = np.zeros((len(docs), len(word2idx)))

    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)
        features = []

        for n in range(ngram_range[0], ngram_range[1]+1):
            for i in range(len(tokens)-n+1):
                features.append(" ".join(tokens[i:i+n]))

        counts = Counter(features)

        for word, count in counts.items():
            if word in word2idx:
                matrix[doc_id, word2idx[word]] = count

    return matrix

def compute_tf(bow):
    tf = bow.copy()
    row_sums = tf.sum(axis=1, keepdims=True)
    tf = tf / (row_sums + 1e-9)
    return tf

def compute_idf(bow):
    N = bow.shape[0]
    df = np.count_nonzero(bow > 0, axis=0)
    idf = np.log(N / (df + 1))
    return idf

def tfidf_matrix(bow):
    tf = compute_tf(bow)
    idf = compute_idf(bow)
    return tf * idf


