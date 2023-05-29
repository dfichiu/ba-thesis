import nltk
#nltk.download('punkt')
from nltk import tokenize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import string

def lowercase_text(text):
    return text.lower()

def split_text_into_sentences(text):
    return tokenize.sent_tokenize(text)

def remove_punctuation(text):
  return [s.translate(str.maketrans({key: "" for key in string.punctuation})) for s in text]

def add_whitespace_before_punctuation(text):
  return [s.translate(str.maketrans({key: " {0}".format(key) for key in string.punctuation})) for s in text]

def tokenize_sentence(text):
  return [nltk.word_tokenize(sentence) for sentence in text]

pipeline = Pipeline([
    ('lowercase-text', FunctionTransformer(lowercase_text)),
    ('tokenize-text-into-sentences', FunctionTransformer(split_text_into_sentences)),
    ('remove-punctuation', FunctionTransformer(remove_punctuation)),
    ('tokenize-sentence', FunctionTransformer(tokenize_sentence)),

])

def preprocess_text(text):
  # TODO: Text is actually the wrong name for the function, since it only preprocesses sentences.
  # Think about function vectorization (split function into two and vectorize one)/loops to make the function preprocess an entire list.
  global pipeline
  pipeline.fit(text)

  return pipeline.transform(text)[0]