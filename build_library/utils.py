import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')


class customNlp:

    def __init__(self):
        self.punct = string.punctuation

    def text_data_cleaning(self, sentence):
        punct = string.punctuation
        doc = nlp(sentence)

        tokens = []
        for token in doc:
            if token.lemma_ != "-PRON-":
                temp = token.lemma_.lower().strip()
            else:
                temp = token.lower_
            tokens.append(temp)

        cleaned_tokens = []
        for token in tokens:
            if token not in stopwords and token not in self.punct:
                cleaned_tokens.append(token)
        return cleaned_tokens
