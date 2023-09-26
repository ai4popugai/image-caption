import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def dot_product_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, scaled: bool = False):
    scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(query.shape[-1] if scaled else 1)
    weights = F.softmax(scores, dim=-1)
    context = torch.matmul(weights, values)
    return context, weights


class BahdanauAttention(nn.Module):
    def __init__(self, query_hidden_size: int, keys_hidden_size: int,
                 out_hidden_size: int,  hidden_size: int = 512):
        """
        Bahdanau attention mechanism

        :param query_hidden_size: hidden size of query
        :param keys_hidden_size: hidden size of keys
        :param hidden_size: inner hidden size and context hidden size
        :param out_hidden_size: hidden size of output hidden context vector
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.query_hidden_size = query_hidden_size
        self.keys_hidden_size = keys_hidden_size
        self.out_hidden_size = out_hidden_size
        self.Wq = nn.Linear(self.query_hidden_size, self.hidden_size, bias=False)
        self.Wk = nn.Linear(self.keys_hidden_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.keys_hidden_size, self.hidden_size, bias=False)
        self.Wout = nn.Linear(self.hidden_size, self.out_hidden_size)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        query, keys, values = self.Wq(query), self.Wk(keys), self.Wv(keys)
        context, weights = dot_product_attention(query, keys, values)
        context = self.Wout(context)
        return context, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, query_hidden_size: int, keys_hidden_size: int,
                 out_hidden_size: int, num_heads: int,  hidden_size: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.d_k = self.hidden_size // self.num_heads
        self.query_hidden_size = query_hidden_size
        self.keys_hidden_size = keys_hidden_size
        self.out_hidden_size = out_hidden_size

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, self.num_heads, seq_length, self.d_k)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor]):
        query = self.split_heads(self.Wq(query))
        keys = self.split_heads(self.Wk(keys))
        values = self.split_heads(self.Wv(values))


if __name__ == '__main__':
    q = torch.rand(8, 1, 128)
    k = torch.rand(8, 5, 256)
    attention = BahdanauAttention(query_hidden_size=128, keys_hidden_size=256,
                                  hidden_size=512, out_hidden_size=1024)
    c, att_w = attention(q, k)
    print(c.shape)  # torch.Size([8, 1, 1024])
    print(att_w.shape)  # torch.Size([8, 1, 5])

