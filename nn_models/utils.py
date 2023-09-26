import math
import time

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.position_indices = torch.arange(self.max_seq_len, dtype=torch.float32).view(1, -1, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32)
                             * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        position_embeddings = torch.zeros(self.max_seq_len, self.d_model)
        position_embeddings[:, 0::2] = torch.sin(self.position_indices * div_term)
        position_embeddings[:, 1::2] = torch.cos(self.position_indices * div_term)
        self.position_embeddings = position_embeddings

    def forward(self, x: torch.Tensor):
        bs, seg_len, depth = x.shape
        if x.shape[-1] != self.d_model:
            raise RuntimeError(f'Could not apply positional encoding to tensor with depth {depth}, '
                               f'only depth={self.d_model} allowed')
        return x + self.position_embeddings.to(x.device).expand(bs, -1, -1)[:, :seg_len, :]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


if __name__ == "__main__":
    # Example usage
    d_model = 512
    max_seq_length = 100

    # Create a random input tensor (batch_size, seq_length, d_model)
    batch_size = 32
    seq_length = 50
    pe = PositionalEncoding(d_model, max_seq_length)
    input_tensor = torch.randn(batch_size, seq_length, d_model).to('mps')

    # Apply positional encoding to the input tensor
    start_time = time.perf_counter()
    output_tensor = pe(input_tensor)
    print(time.perf_counter() - start_time)

    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)
