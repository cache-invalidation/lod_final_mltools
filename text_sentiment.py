from enum import Enum

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

dostoevsky_tokenizer = RegexTokenizer()
dostoevsky_model = FastTextSocialNetworkModel(tokenizer=dostoevsky_tokenizer)

class Sentiment(Enum):
    POSITIVE = 1
    NEUTRAL = 2
    NEGATIVE = 3
    UNIDENTIFIED = 4

def process_label(label: str) -> Sentiment:
    if label == 'positive':
        return Sentiment.POSITIVE
    elif label == 'negative':
        return Sentiment.NEGATIVE
    elif label == 'neutral' or label == 'speech':
        return Sentiment.NEUTRAL
    else:
        return Sentiment.UNIDENTIFIED

def get_texts_sentiment(texts: list[str]) -> list[Sentiment]:
    estimates = dostoevsky_model.predict(texts, k=1)
    labels = map(lambda x: list(x.keys())[0], estimates)
    labels = map(process_label, labels)

    return list(labels)

def get_text_sentiment(text: str) -> Sentiment:
    return get_texts_sentiment([text])[0]
