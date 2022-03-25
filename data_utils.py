# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd

def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = 'glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    def __init__(self, dataset='rest15', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'rest15': {
                'train': './data/restaurant15/sem15_restaurant_train.csv',
                'test': './data/restaurant15/sem15_restaurant_test.csv'
            },
            'lap15': {
                'train': './data/Laptop15/sem15_laptops_train.csv',
                'test': './data/Laptop15/sem15_laptops_test.csv'
            },
            'rest16': {
                'train': './data/restaurant16/sem16_restaurant_train.csv',
                'test': './data/restaurant16/sem16_restaurant_test.csv'
            },
            'lap16': {
                'train': './data/Laptop16/sem16_laptops_train.csv',
                'test': './data/Laptop16/sem16_laptops_test.csv'
            }
        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset + '_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset + '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)

        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))

    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = pd.read_csv(fname, encoding='gbk')
            fins = fin[['content', 'aspect', 'label', 'sentiment_value']]
            for i in range(0, len(fins)):
                text += fin['content'][i] + " "
        return text

    def __read_data__(fname, tokenizer):
        fin = pd.read_csv(fname, encoding='gbk')
        fins = fin[['content', 'aspect', 'label', 'sentiment_value']]
        fin_graph = open(fname + '.graph', 'rb')
        idx2graph = pickle.load(fin_graph)
        fin_graph.close()
        fin_tree = open(fname + '.tree', 'rb')
        idx2tree = pickle.load(fin_tree)
        fin_tree.close()

        all_data = []
        for i in range(0, len(fins)):
            text_indices = tokenizer.text_to_sequence(fins['content'][i])
            aspect_indices = tokenizer.text_to_sequence(fins['aspect'][i])
            dependency_graph = idx2graph[i]
            dependency_tree = idx2tree[i]
            aspect_label = int(fins['label'][i])
            polarity = fins['sentiment_value'][i] + 1
            data = {
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'aspect_label': aspect_label,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'dependency_tree': dependency_tree,
            }
            all_data.append(data)
        return all_data