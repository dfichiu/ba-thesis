from nltk import ngrams, tokenize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import string


text = "Text chunking is an intermediate step towards full parsing. It was the shared task for CoNLL-2000. Training and test data for this task is available. This data consists of the same partitions of the Wall Street Journal corpus (WSJ) as the widely used data for noun phrase chunking: sections 15-18 as training data (211727 tokens) and section 20 as test data (47377 tokens). The annotation of the data has been derived from the WSJ corpus by a program written by Sabine Buchholz from Tilburg University, The Netherlands."


def lowercase_text(text):
    return text.lower()

def split_text_into_sentences(text):
    return tokenize.sent_tokenize(text)

def add_whitespace_before_punctuation(text):
    return [sentence.translate(str.maketrans({key: " {0}".format(key) for key in string.punctuation})) for sentence in text]

def construct_ngrams(text):
    pass


pipeline = Pipeline([
    ('lowercase-text', FunctionTransformer(lowercase_text)),
    ('tokenize-text-into-sentences', FunctionTransformer(split_text_into_sentences)),
    ('add-whitespace-before-punctuation', FunctionTransformer(add_whitespace_before_punctuation)),

]) 

pipeline.fit(text)

preprocessed_text = pipeline.transform(text)
print(preprocessed_text)
#print(string.punctuation)