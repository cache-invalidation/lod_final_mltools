from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

from .sentiment import Sentiment

dostoevsky_tokenizer = RegexTokenizer()
dostoevsky_model = FastTextSocialNetworkModel(tokenizer=dostoevsky_tokenizer)

def process_label(label: str) -> Sentiment:
    """
    Turn text label into instance of `Sentiment` class

    Arguments:
    ----------
        label: str
            Original label provided by the model

    Returns:
    --------
        Corresponding instance of `Sentiment` class
    """
    if label == 'positive':
        return Sentiment.POSITIVE
    elif label == 'negative':
        return Sentiment.NEGATIVE
    else:
        return Sentiment.NEUTRAL

def get_texts_sentiment(texts: list[str]) -> list[Sentiment]:
    """
    Make sentiment predictions for texts

    Arguments:
    ----------
        texts: list[str]
            List of texts to predict sentiment for
    
    Returns:
    --------
        list[Sentiment]
            Predictions for sentiments
    """
    estimates = dostoevsky_model.predict(texts, k=1)
    labels = map(lambda x: list(x.keys())[0], estimates)
    labels = map(process_label, labels)

    return list(labels)

def get_text_sentiment(text: str) -> Sentiment:
    """
    Make sentiment prediction for single text

    Arguments:
    ----------
        text: str
            Text to predict sentiment for

    Returns:
    --------
        Sentiment
            Prediction for sentiment
    """
    return get_texts_sentiment([text])[0]
