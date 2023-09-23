import torch
from torch import nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, query_hidden_size: int, keys_hidden_size: int, hidden_size: int = 512):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.Wa = nn.Linear(query_hidden_size, self.hidden_size)
        self.Ua = nn.Linear(keys_hidden_size, self.hidden_size)
        self.Va = nn.Linear(self.hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights


if __name__ == '__main__':
    query = torch.rand(8, 1, 128)
    keys = torch.rand(8, 5, 256)
    attention = BahdanauAttention(128, 256)
    context, weights = attention(query, keys)
    print(context.shape)  # torch.Size([8, 1, 256])
    print(weights.shape)  # torch.Size([8, 1, 5])

