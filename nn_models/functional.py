import torch
from torch import nn
import torch.nn.functional as F


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
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query_hidden_size = query_hidden_size
        self.keys_hidden_size = keys_hidden_size
        self.out_hidden_size = out_hidden_size
        self.Wq = nn.Linear(self.query_hidden_size, self.hidden_size, bias=False)
        self.Wk = nn.Linear(self.keys_hidden_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.keys_hidden_size, self.hidden_size, bias=False)
        self.Wout = nn.Linear(self.hidden_size, self.out_hidden_size)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        scores = torch.matmul(self.Wq(query), self.Wk(keys).transpose(-1, -2))
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, self.Wv(keys))
        context = self.Wout(context)
        return context, weights


if __name__ == '__main__':
    query = torch.rand(8, 1, 128)
    keys = torch.rand(8, 5, 256)
    attention = BahdanauAttention(query_hidden_size=128, keys_hidden_size=256,
                                  hidden_size=512, out_hidden_size=1024)
    context, weights = attention(query, keys)
    print(context.shape)  # torch.Size([8, 1, 1024])
    print(weights.shape)  # torch.Size([8, 1, 5])

