# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size)) #迭代次数
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_aspect_indices = []
        batch_dependency_graph = []
        batch_aspect_label = []
        batch_polarity = []
        batch_dependency_tree = []


        max_len = max([len(t[self.sort_key]) for t in batch_data])

        for item in batch_data:
            text_indices = item['text_indices']
            aspect_indices = item['aspect_indices']
            dependency_graph = item['dependency_graph']
            dependency_tree = item['dependency_tree']
            aspect_label = item['aspect_label']
            polarity = item['polarity']

            text_padding = [0] * (max_len - len(text_indices))
            #aspect_padding = [0] * (max_len - len(aspect_indices))
            batch_dependency_graph.append(numpy.pad(dependency_graph, ((0, max_len-len(text_indices)),
                                                                       (0, max_len-len(text_indices))), 'constant'))
            batch_dependency_tree.append(numpy.pad(dependency_tree, ((0, max_len - len(text_indices)),
                                                                     (0, max_len - len(text_indices))), 'constant'))
            batch_text_indices.append(text_indices + text_padding)
            batch_aspect_indices.append(aspect_indices)
            batch_aspect_label.append(aspect_label)
            batch_polarity.append(polarity)

        return {
            'text_indices': torch.tensor(batch_text_indices),
            'aspect_indices': torch.tensor(batch_aspect_indices),
            'dependency_graph': torch.tensor(batch_dependency_graph),
            'dependency_tree': torch.tensor(batch_dependency_tree),
            'aspect_label': torch.tensor(batch_aspect_label),
            'polarity': torch.tensor(batch_polarity)
             }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]


