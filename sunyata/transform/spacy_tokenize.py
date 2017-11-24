import spacy
from spacy.tokenizer import Tokenizer

from .base import Transformer


class SpacyTokenize(Transformer):
    def __init__(self):
        nlp = spacy.load('en')
        self.tok = Tokenizer(nlp.vocab)

    def transform(self, xx):
        rrr = []
        for doc in self.tok.pipe(xx):
            rr = []
            for token in doc:
                rr.append(token.text.lower())
            rrr.append(rr)
        return rrr
