# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class ATLSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATLSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_embed_dropout = nn.Dropout(0.3)
        self.text_lstm = DynamicLSTM(2 * opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.aspect_fc = nn.Linear(3 * opt.hidden_dim, opt.aspect_polarity)

        self.weight_W = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 1))
        self.weight_v = nn.Parameter(torch.Tensor(opt.hidden_dim, 2 * opt.hidden_dim))
        self.weight_r = nn.Parameter(torch.Tensor(3 * opt.hidden_dim, 3 * opt.hidden_dim))
        # self.weight_p = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)
        nn.init.uniform_(self.weight_v, -0.1, 0.1)
        nn.init.uniform_(self.weight_r, -0.1, 0.1)
        # nn.init.uniform_(self.weight_p, -0.1, 0.1)

    def forward(self, inputs):
        text_indices, aspect_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        text = self.embed(text_indices)
        aspect = self.embed(aspect_indices)

        text_aspect = torch.cat([text, aspect * torch.ones_like(text)], -1)
        text_aspect = self.text_embed_dropout(text_aspect)
        hidden, (_, _) = self.text_lstm(text_aspect, text_len)

        #attention实验1
        u = torch.tanh(torch.matmul(hidden, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        alpha = F.softmax(att, dim=1).transpose(1, 2)

        #attention实验2
        # u = torch.tanh(torch.cat([torch.matmul(hidden, self.weight_W), torch.matmul(aspect, self.weight_v) * torch.ones_like(hidden)], -1))
        # att = torch.matmul(u, self.weight_proj)
        # alpha = F.softmax(att, dim=1).transpose(1, 2)


        # as_h = F.tanh(torch.cat([torch.matmul(alpha, hidden), aspect], -1)).squeeze(1)
        as_h = F.tanh(torch.matmul(torch.cat([torch.matmul(alpha, hidden), aspect], -1), self.weight_r)).squeeze(1)
        # as_h = F.tanh(torch.matmul(torch.cat([torch.matmul(alpha, hidden), aspect], -1), self.weight_r) + torch.matmul(torch.cat([hidden[:, -1, :].unsqueeze(1), aspect], -1), self.weight_p)).squeeze(1)
        # as_h = F.tanh(torch.cat([torch.matmul(alpha, hidden), aspect], -1) + torch.cat([hidden[:, -1, :].unsqueeze(1), aspect], -1)).squeeze(1)
        # as_h = F.tanh(torch.matmul(torch.matmul(alpha, hidden), self.weight_r) + torch.matmul(hidden[:, -1, :].unsqueeze(1), self.weight_p)).squeeze(1)
        aspect_output = F.sigmoid(self.aspect_fc(as_h))

        return aspect_output, alpha
