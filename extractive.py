from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def extractive_summary(text, n_sentences=3):
    """
    Реализует экстрактивную суммаризацию с использованием TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text)

    # Сумма TF-IDF для каждого предложения
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    ranked_sentences = [text[i] for i in np.argsort(-sentence_scores)]

    return ranked_sentences[:n_sentences]
