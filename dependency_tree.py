# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import spacy
import pickle
from spacy.tokens import Doc


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1

    return matrix


def process(filename):
    fin = pd.read_csv(filename, encoding='gbk')
    fin = fin[['content', 'aspect', 'label', 'sentiment_value']]
    idx2graph = {}
    fout = open(filename+'.tree', 'wb')
    for i in range(0, len(fin)):
        adj_matrix = dependency_adj_matrix(fin['content'][i])
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == '__main__':
    # process('./data/restaurant15/sem15_restaurant_train.csv')
    # process('./data/restaurant15/sem15_restaurant_test.csv')
    process('./data/restaurant16/sem16_restaurant_train.csv')
    process('./data/restaurant16/sem16_restaurant_test.csv')
    # process('./data/Laptop16/sem16_laptops_train.csv')
    # process('./data/Laptop16/sem16_laptops_test.csv')
    # process('./data/Laptop15/sem15_laptops_train.csv')
    # process('./data/Laptop15/sem15_laptops_test.csv')