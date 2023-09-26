import math
import time

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.position_indices = torch.arange(self.max_seq_len, dtype=torch.float32).view(1, -1, 1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, dtype=torch.float32)
                             * -(torch.log(torch.tensor(10000.0)) / self.hidden_size))
        position_embeddings = torch.zeros(self.max_seq_len, self.hidden_size)
        position_embeddings[:, 0::2] = torch.sin(self.position_indices * div_term)
        position_embeddings[:, 1::2] = torch.cos(self.position_indices * div_term)
        self.position_embeddings = position_embeddings

    def forward(self, x: torch.Tensor):
        bs, seg_len, depth = x.shape
        if x.shape[-1] != self.hidden_size:
            raise RuntimeError(f'Could not apply positional encoding to tensor with depth {depth}, '
                               f'only depth={self.hidden_size} allowed')
        return x + self.position_embeddings.to(x.device)[:seg_len, :]


if __name__ == "__main__":
    # Example usage
    hs = 512
    max_seq_length = 100

    # Create a random input tensor (batch_size, seq_length, hidden_size)
    batch_size = 32
    seq_length = 50
    pe = PositionalEncoding(hs, max_seq_length)
    input_tensor = torch.randn(batch_size, seq_length, hs)

    # Apply positional encoding to the input tensor
    start_time = time.perf_counter()
    output_tensor = pe(input_tensor)
    print(time.perf_counter() - start_time)

    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)
