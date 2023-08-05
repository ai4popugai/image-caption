import torch
from torch import nn

from models.functional import BahdanauAttention


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, keys_hidden_size: int,
                 sos_tokenized: torch.Tensor, eos_tokenized: torch.Tensor,
                 num_layers=4, max_len=15,):
        super(AttentionDecoder, self).__init__()
        # attention mechanism for feature maps and hidden state
        self.attention = BahdanauAttention(hidden_size, keys_hidden_size)
        # linear layer to transform context vector from keys_hidden_size to hidden_size
        self.fc_0 = nn.Linear(keys_hidden_size, hidden_size)
        # linear layer to transform input embeddings from embedding_size to hidden_size
        self.fc_1 = nn.Linear(embedding_size, hidden_size)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # embedding layer to convert words indexex to embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # rnn cell
        self.rnn = nn.GRU(hidden_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        # linear layer to make final predictions
        self.fc_2 = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len
        self.sos_tokenized = sos_tokenized
        self.eos_tokenized = eos_tokenized

    def _tokens_to_x(self, tokenized_words: torch.Tensor) -> torch.Tensor:
        """
        Convert tokenized words firstly to embeddings and then to x.

        :param tokenized_words: (batch_size, seq_len) - tokenized words.
        :return: (batch_size, seq_len, hidden_size) - ready to go x.
        """
        embeddings = self.embedding(tokenized_words)  # embeddings: (batch_size, seq_len, embedding_size)
        return self.fc_1(embeddings)  # x: (batch_size, seq_len, hidden_size) - ready to go captions

    def _forward(self, x_t, hidden, keys) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        One step forward of the decoder, given the input x_t, the hidden state and the keys (feature maps).
        x_t is the word ready to go vector of the previous gt word if training mode or previous predicted word
        if inference mode,
        hidden is the hidden state of the previous time step.

        :param x_t: ready to go vector of the previous gt word or previous predicted word (batch_size, 1, hidden_size).
        :param hidden: hidden state of the previous time step (batch_size, hidden_size).
        :param keys: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :return: decoded word (batch_size, vocab_size), hidden state (batch_size, hidden_size) and attention weights.
        """
        context, weights = self.attention(hidden.unsqueeze(1), keys)  # context: (batch_size, 1, keys_hidden_size)
                                                                      # weights: (batch_size, 1, seq_len)
        context = self.fc_0(context)  # context: (batch_size, 1, hidden_size)
        rnn_input = torch.cat([x_t, context], dim=-1)  # rnn_input: (batch_size, 1, hidden_size + hidden_size)
        # GRU takes input of shape (batch_size, seq_len, input_size) and hidden of shape (1, batch_size, hidden_size)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))   # output: (batch_size, 1, hidden_size)
                                                                    # hidden: (batch_size, hidden_size)
        output = output.squeeze(1)  # output: (batch_size, hidden_size)
        output = self.fc_2(output)  # output: (batch_size, vocab_size)
        return output, hidden.squeeze(0), weights

    def _forward_inference(self, features) -> torch.Tensor:
        """
        Forward pass in inference mode. Captions are not provided.
        Decode the sequence step by step, until the EOS token is predicted or the max_len is reached.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :return: (batch_size, any_seq_len, vocab_size) - predictions for each token in the sequence.
        """

        batch_size = features.shape[0]
        hidden = torch.zeros(batch_size, self.hidden_size, device=features.device)  # hidden: (batch_size, hidden_size)
        # outputs: (batch_size, max_len, vocab_size)
        outputs = torch.zeros(batch_size, self.max_len, self.vocab_size, device=features.device)

        # initialize the input with SOS token
        x_t = self._tokens_to_x(self.sos_tokenized).repeat(batch_size, 1)  # x_t: (batch_size, 1, hidden_size)

        for t in range(self.max_len):
            output, hidden, _ = self._forward(x_t, hidden, features)  # output: (batch_size, vocab_size)
                                                                      # hidden: (batch_size, hidden_size)
            outputs[:, t, :] = output
            # get the most probable word index
            decoded_token = output.argmax(dim=-1)  # x_t: (batch_size,)
            # check if x_t is EOS token
            if (decoded_token == self.eos_tokenized).all().item():
                return outputs[:, :t + 1, :]
            # get the embedding of the predicted word
            x_t = self._tokens_to_x(decoded_token.unsqueeze(1))  # x_t: (batch_size, 1, hidden_size)
        return outputs

    def _forward_training(self, features, captions) -> torch.Tensor:
        """
        Forward pass in training mode. Captions are provided.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :param captions: captions to train on (batch_size, seq_len).
        :return: (batch_size, seq_len - 1, vocab_size) - predictions for each token in the sequence except EOS token.
        """
        out_seq_len = captions.shape[1] - 1  # -1 because we don't need to predict EOS token
        x = self._tokens_to_x(captions)  # x: (batch_size, seq_len, hidden_size) - ready to go captions
        batch_size = features.shape[0]
        hidden = torch.zeros(batch_size, self.hidden_size, device=features.device)  # hidden: (batch_size, hidden_size)
        # outputs: (batch_size, seq_len, vocab_size)
        outputs = torch.zeros(batch_size, out_seq_len, self.vocab_size, device=features.device)
        for t in range(out_seq_len):
            x_t = x.select(1, t).unsqueeze(1)  # embedding: (batch_size, 1, hidden_size)
            output, hidden, _ = self._forward(x_t, hidden, features)
            outputs[:, t, :] = output
        return outputs

    def forward(self, features, captions: torch.Tensor = None) -> torch.Tensor:
        """
        The main method of the decoder. If captions are provided, the method will run in training mode.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :param captions: captions to train on (batch_size, seq_len).
        :return: output predictions (batch_size, seq_len - 1, vocab_size) if captions are provided,
        else (batch_size, any_seq_len, vocab_size)
        """
        if captions is not None:
            return self._forward_training(features, captions)  # (batch_size, seq_len, vocab_size)
        else:
            # in that case sequence length is not known
            return self._forward_inference(features)  # (batch_size, any_seq_len <= self.max_len, vocab_size)
