import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk import tokenize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from typing import List

import string

def split_text_into_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)

def lowercase(sentences):
    return [s.lower() for s in sentences]

def remove_punctuation(sentences):
    return [s.translate(str.maketrans({key: "" for key in string.punctuation})) for s in sentences]

def remove_stopwords(sentences_tokens):
    return [[token for token in sentence_tokens if token not in stopwords.words('english')] for sentence_tokens in sentences_tokens]

def add_whitespace_before_punctuation(text):
    return [s.translate(str.maketrans({key: " {0}".format(key) for key in string.punctuation})) for s in text]

def tokenize_sentence(sentences):
    return [nltk.word_tokenize(s) for s in sentences]

pipeline = Pipeline([
    ('tokenize-text-into-sentences', FunctionTransformer(split_text_into_sentences)),
    ('lowercase', FunctionTransformer(lowercase)),
    ('remove-punctuation', FunctionTransformer(remove_punctuation)),
    ('tokenize-sentence', FunctionTransformer(tokenize_sentence)),
    ('remove-stopwords', FunctionTransformer(remove_stopwords))
])

def preprocess_text(text: str) -> List[str]:
    """
    """
    global pipeline
    pipeline.fit(text)

    return pipeline.transform(text)
