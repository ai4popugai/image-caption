import torch
from torch import nn

from models.functional import BahdanauAttention


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, keys_hidden_size: int, num_layers=4,
                 max_len=15):
        super(AttentionDecoder, self).__init__()
        # attention mechanism for feature maps and hidden state
        self.attention = BahdanauAttention(hidden_size, keys_hidden_size)
        # linear layer to transform context vector from keys_hidden_size to hidden_size
        self.fc_0 = nn.Linear(keys_hidden_size, hidden_size)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # embedding layer to convert words indexex to embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # rnn cell
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        # linear layer to make final predictions
        self.fc_1 = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len

    def _forward(self, embedding, hidden, keys):
        # embedding: (batch_size, 1, embedding_size)
        # hidden: (batch_size, hidden_size)
        # keys: (batch_size, seq_len, keys_hidden_size)
        context, weights = self.attention(hidden.unsqueeze(1), keys)  # context: (batch_size, 1, keys_hidden_size)
                                                                      # weights: (batch_size, 1, seq_len)
        context = self.fc_0(context)  # context: (batch_size, 1, hidden_size)
        rnn_input = torch.cat([embedding, context], dim=-1)  # rnn_input: (batch_size, 1, embedding_size + hidden_size)
        # GRU takes input of shape (batch_size, seq_len, input_size) and hidden of shape (1, batch_size, hidden_size)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))   # output: (batch_size, 1, hidden_size)
                                                                    # hidden: (batch_size, hidden_size)
        output = output.squeeze(1)  # output: (batch_size, hidden_size)
        output = self.fc_1(output)  # output: (batch_size, vocab_size)
        return output, hidden.squeeze(0), weights

    def forward(self, features, captions):
        # features: (batch_size, seq_len, keys_hidden_size)
        # captions: (batch_size, seq_len)
        embeddings = self.embedding(captions)  # embeddings: (batch_size, seq_len, embedding_size)
        batch_size = features.shape[0]
        hidden = torch.zeros(batch_size, self.hidden_size, device=features.device)
        outputs = torch.zeros(batch_size, self.max_len, self.vocab_size, device=features.device)
        for t in range(self.max_len):
            embedding = embeddings.select(1, t).unsqueeze(1)  # embedding: (batch_size, 1, embedding_size)
            output, hidden, _ = self._forward(embedding, hidden, features)
            outputs[:, t] = output
        return outputs
