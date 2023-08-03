from torch import nn

from models.functional import BahdanauAttention


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, keys_hidden_size: int):
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
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size)
        # linear layer to make final predictions
        self.fc_1 = nn.Linear(hidden_size, vocab_size)
