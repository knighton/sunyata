import spacy

from .base import Transformer


class Tokenize(Transformer):
    def __init__(self):
        self.nlp = spacy.load('en')

    def transform(self, xx):
        rrr = []
        for x in xx:
            doc = self.nlp(x)
            rrr.append(list(doc))
        return rrr
