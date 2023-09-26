import math
import time

import torch
from torch import nn


def positional_encoding(x: torch.Tensor):
    bs, seq_len, depth = x.shape
    position_indices = torch.arange(seq_len, dtype=torch.float32, device=x.device).view(1, -1, 1)
    div_term = torch.exp(torch.arange(0, depth, 2, dtype=torch.float32, device=x.device)
                         * -(torch.log(torch.tensor(10000.0, device=x.device)) / depth))
    position_embeddings = torch.zeros(seq_len, depth, device=x.device)

    position_embeddings[:, 0::2] = torch.sin(position_indices * div_term)
    position_embeddings[:, 1::2] = torch.cos(position_indices * div_term)
    position_embeddings = position_embeddings.expand(bs, -1, -1)

    return x + position_embeddings


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
    input_tensor = torch.randn(batch_size, seq_length, d_model).to('mps')

    # Apply positional encoding to the input tensor
    start_time = time.perf_counter()
    output_tensor = positional_encoding(input_tensor)
    print(time.perf_counter() - start_time)

    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)
