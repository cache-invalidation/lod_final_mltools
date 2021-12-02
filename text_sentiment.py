from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

from .sentiment import Sentiment

dostoevsky_tokenizer = RegexTokenizer()
dostoevsky_model = FastTextSocialNetworkModel(tokenizer=dostoevsky_tokenizer)

def process_label(label: str) -> Sentiment:
    if label == 'positive':
        return Sentiment.POSITIVE
    elif label == 'negative':
        return Sentiment.NEGATIVE
    else:
        return Sentiment.NEUTRAL

def get_texts_sentiment(texts: list[str]) -> list[Sentiment]:
    estimates = dostoevsky_model.predict(texts, k=1)
    labels = map(lambda x: list(x.keys())[0], estimates)
    labels = map(process_label, labels)

    return list(labels)

def get_text_sentiment(text: str) -> Sentiment:
    return get_texts_sentiment([text])[0]
