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

class Attention_Layer(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)

    def forward(self, H_GCN, aspect_term, aspect_location):

        # 计算生成QKV矩阵
        Q = self.Q_linear(aspect_term)
        K = self.K_linear(H_GCN).permute(0, 2, 1)  # 先进行一次转置
        V = self.K_linear(H_GCN)

        # 下面开始计算啦, mask
        alpha_mat = torch.matmul(Q, K)
        alpha_mat_j = alpha_mat.squeeze(1)
        alpha_mat_j[torch.arange(aspect_location.shape[0]), aspect_location[:, 0]] = 0
        alpha_mask = alpha_mat_j.unsqueeze(1)

        # 下面开始softmax
        alpha = F.softmax(alpha_mask, dim=2)
        #out = torch.matmul(alpha, V)

        return alpha

class ATLSGCNDT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATLSGCNDT, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_embed_dropout = nn.Dropout(0.3)
        self.text_lstm = DynamicLSTM(2 * opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.emo_att = Attention_Layer(opt.hidden_dim)

        self.aspect_fc = nn.Linear(3 * opt.hidden_dim, opt.aspect_polarity)
        self.softmax = nn.LogSoftmax(dim=1)
        self.emotion_fc = nn.Linear(2 * opt.hidden_dim, opt.emotion_polarity)

        # 实验
        self.weight_W = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 1))
        self.weight_r = nn.Parameter(torch.Tensor(3 * opt.hidden_dim, 3 * opt.hidden_dim))
        self.weight_mask = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)
        nn.init.uniform_(self.weight_r, -0.1, 0.1)
        nn.init.uniform_(self.weight_mask, -0.1, 0.1)

    def forward(self, inputs):
        text_indices, aspect_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)

        text = self.embed(text_indices)
        aspect = self.embed(aspect_indices)
        text_aspect = torch.cat([text, aspect * torch.ones_like(text)], -1)
        text_aspect = self.text_embed_dropout(text_aspect)
        hidden, (hidden_last, _) = self.text_lstm(text_aspect, text_len)

        # aspect
        u = torch.tanh(torch.matmul(hidden, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        alpha = F.softmax(att, dim=1).transpose(1, 2)
        as_h = F.tanh(torch.matmul(torch.cat([torch.matmul(alpha, hidden), aspect], -1), self.weight_r)).squeeze(1)
        aspect_output = F.sigmoid(self.aspect_fc(as_h))

        # emotion
        gch_hidden_two = F.relu(self.gc1(hidden, adj))

        # gch_hidden_one = F.relu(self.gc1(hidden, adj))
        # gch_hidden_two = F.relu(self.gc1(gch_hidden_one, adj))

        # gch_hidden_one = F.relu(self.gc1(hidden, adj))
        # gch_hidden_two_s = F.relu(self.gc1(gch_hidden_one, adj))
        # gch_hidden_two = F.relu(self.gc1(gch_hidden_two_s, adj))

        # gch_hidden_one = F.relu(self.gc1(hidden, adj))
        # gch_hidden_two_s = F.relu(self.gc1(gch_hidden_one, adj))
        # gch_hidden_two_t = F.relu(self.gc1(gch_hidden_two_s, adj))
        # gch_hidden_two = F.relu(self.gc1(gch_hidden_two_t, adj))

        # emotion aspect
        aspect_term_location = torch.argmax(alpha, dim=2)
        aspect_term = gch_hidden_two[(torch.arange(aspect_term_location.shape[0]), aspect_term_location[:, 0])].unsqueeze(1)
        emotion_attention = self.emo_att(gch_hidden_two, aspect_term, aspect_term_location)
        GCN_hidden_attention = torch.matmul(emotion_attention, gch_hidden_two)
        p = F.tanh(torch.matmul(torch.sum(GCN_hidden_attention, dim=1), self.weight_mask))
        emotion_output = self.softmax(self.emotion_fc(p))

        return aspect_output, emotion_output, alpha, emotion_attention









