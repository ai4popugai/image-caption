from torch import nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, d_ff)
        self.fc2 = nn.Linear(d_ff, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

