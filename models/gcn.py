# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_embed_dropout = nn.Dropout(0.3)
        self.text_lstm = DynamicLSTM(2 * opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                     bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)

        self.aspect_fc = nn.Linear(3 * opt.hidden_dim, opt.aspect_polarity)
        self.softmax = nn.LogSoftmax(dim=1)
        self.emotion_fc = nn.Linear(2 * opt.hidden_dim, opt.emotion_polarity)

        # 实验
        self.weight_W = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 1))
        self.weight_r = nn.Parameter(torch.Tensor(3 * opt.hidden_dim, 3 * opt.hidden_dim))
        self.weight_mask = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        # self.weight_mask = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 1))
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)
        nn.init.uniform_(self.weight_r, -0.1, 0.1)
        nn.init.uniform_(self.weight_mask, -0.1, 0.1)

    def position_weight(self, x, alpha):
        hidden_alpha = torch.mul(x, alpha.transpose(1, 2))
        return hidden_alpha

    def mask(self, x, alpha_x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        alpha = alpha_x.squeeze(1).detach().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if alpha[i][j] < np.mean(alpha[i]):
                    mask[i].append(0)
                else:
                    mask[i].append(1)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        #aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        text = self.embed(text_indices)
        aspect = self.embed(aspect_indices)
        text_aspect = torch.cat([text, aspect * torch.ones_like(text)], -1)
        text_aspect = self.text_embed_dropout(text_aspect)
        hidden, (_, _) = self.text_lstm(text_aspect, text_len)
        #attantion

        # alpha_mat = torch.tanh(hidden).transpose(1, 2)
        # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        # as_h = F.tanh(torch.cat([torch.matmul(alpha, hidden), aspect], -1)).squeeze(1)

        # attantion实验1
        u = torch.tanh(torch.matmul(hidden, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        alpha = F.softmax(att, dim=1).transpose(1, 2)

        gch_hidden_one = F.relu(self.gc1(self.position_weight(hidden, alpha), adj))
        gch_hidden_two = F.relu(self.gc1(self.position_weight(gch_hidden_one, alpha), adj))
        hidden_mask = self.mask(gch_hidden_two, alpha)

        # emotion_output = self.emotion_fc(torch.matmul(alpha, hidden_mask).squeeze(1))
        p = torch.matmul(torch.sum(hidden_mask, dim=1), self.weight_mask)
        emotion_output = self.softmax(self.emotion_fc(p))

        # p = torch.matmul(hidden_mask, self.weight_mask).squeeze(2)
        # emotion_output = self.softmax(self.emotion_fc(p))

        return emotion_output, alpha
