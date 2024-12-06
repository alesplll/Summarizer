import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


def preprocess_text(text):
    """
    Очищает и токенизирует текст.
    """
    # Удаление HTML-тегов и спецсимволов
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)

    # Приведение к нижнему регистру
    text = text.lower()

    # Токенизация
    sentences = sent_tokenize(text)
    return sentences
