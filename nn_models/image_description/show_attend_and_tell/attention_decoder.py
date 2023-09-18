from typing import Dict

import torch
from torch import nn

from datasets import LOGITS_KEY
from nn_models.functional import BahdanauAttention


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, keys_hidden_size: int,
                 sos_tokenized: torch.Tensor, eos_tokenized: torch.Tensor,
                 num_layers=4, max_len=15, ):
        super().__init__()
        # attention mechanism for feature maps and hidden state
        self.attention = BahdanauAttention(hidden_size, keys_hidden_size)

        # linear layer to transform context vector from keys_hidden_size to hidden_size
        self.fc_0 = nn.Linear(keys_hidden_size, hidden_size)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # embedding layer to convert words indices to embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # linear layer to transform input embeddings from embedding_size to hidden_size
        self.fc_1 = nn.Linear(embedding_size, hidden_size)

        # rnn cell
        self.num_layers = num_layers

        self.rnn = nn.GRU(hidden_size + hidden_size, hidden_size, num_layers=self.num_layers, batch_first=True)

        # linear layer to make final predictions
        self.fc_2 = nn.Linear(hidden_size, vocab_size)

        self.max_len = max_len
        self.sos_tokenized = sos_tokenized
        self.eos_tokenized = eos_tokenized

    def _tokenized_to_hidden(self, tokenized_word: torch.Tensor) -> torch.Tensor:
        """
        Convert tokenized word firstly to embeddings and then to s_hidden.

        :param tokenized_word: (batch_size,) - tokenized words.
        :return: (batch_size, hidden_size) - vector that can be treated as s_hidden.
        """
        embeddings = self.embedding(tokenized_word)  # embeddings: (batch_size, embedding_size)
        return self.fc_1(embeddings)  # x: (batch_size, hidden_size)

    def _forward(self, decoder_input: torch.Tensor, s_hidden: torch.Tensor, keys: torch.Tensor,) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        One step forward of the decoder, given the hidden state s_hidden and the keys (feature maps).
        s_hidden is the hidden state of the previous time step.

        :param decoder_input: (batch_size, hidden_size) tensor from input ground true y_i-1 or previous predicted word.
        :param s_hidden: hidden state of the previous time step (num_layers, batch_size, hidden_size).
        :param keys: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :return: decoded word (batch_size, vocab_size), hidden state (batch_size, hidden_size) and attention weights.
        """
        # I get hidden state (batch_size, 1, hidden_size) from the last layer of GRU to the attention layer
        context, weights = self.attention(s_hidden[-1].unsqueeze(1), keys)  # context: (batch_size, 1, keys_hidden_size)
        # weights: (batch_size, 1, seq_len)

        context = self.fc_0(context)  # context: (batch_size, 1, hidden_size)

        # rnn_input: (batch_size, 1, hidden_size + hidden_size)
        rnn_input = torch.cat([decoder_input.unsqueeze(1), context], dim=-1)

        # GRU takes input of shape (batch_size, 1, 2 * hidden_size
        # and hidden of shape (num_layers, batch_size, hidden_size)
        y_t, s_hidden = self.rnn(rnn_input, s_hidden)  # output: (batch_size, 1, hidden_size)
        # s_hidden: (num_layers, batch_size, hidden_size)

        y_t = y_t.squeeze(1)  # output: (batch_size, hidden_size)
        y_t = self.fc_2(y_t)  # output: (batch_size, vocab_size)
        return y_t, s_hidden, weights

    def _forward_inference(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in inference mode. Captions are not provided.
        Decode the sequence step by step, until the EOS token is predicted or the max_len is reached.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :return: (batch_size, any_seq_len, vocab_size) - predictions for each token in the sequence.
        """

        batch_size = features.shape[0]
        assert batch_size == 1, "In inference mode batch size must be 1."
        # s_hidden: (num_layers, batch_size, hidden_size)
        s_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device)

        # decoder_input: (batch_size, hidden_size), always start with SOS token
        decoder_input = self._tokenized_to_hidden(self.sos_tokenized.unsqueeze(0).expand(batch_size, -1))\
            .to(features.device)

        # outputs: (batch_size, max_len, vocab_size)
        outputs = torch.zeros(batch_size, self.max_len, self.vocab_size, device=features.device)

        for t in range(self.max_len):
            y_t, s_hidden, _ = self._forward(decoder_input, s_hidden, features)  # y_t: (batch_size, vocab_size)
            # s_hidden: (num_layers, batch_size, hidden_size)

            outputs[:, t, :] = y_t

            # get the most probable word index
            decoded_token = y_t.argmax(dim=-1)  # decoded_token: (batch_size,)

            # check if decoded_token is EOS token
            if (decoded_token == self.eos_tokenized).all().item():
                return outputs[:, :t + 1, :]

            # set new decoder_input
            decoder_input = self._tokenized_to_hidden(decoded_token)

        return outputs

    def _forward_training(self, features: torch.Tensor, captions: torch.Tensor, teacher_forcing: bool) -> torch.Tensor:
        """
        Forward pass in training mode. Captions are provided.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :param captions: captions to train on (batch_size, seq_len).
        :param teacher_forcing: if True, train with teacher forcing.
        :return: (batch_size, seq_len - 1, vocab_size) - predictions for each token in the sequence except EOS token.
        """
        captions = captions[:, 1:]  # because we don't need to predict SOS token
        out_seq_len = captions.shape[1]
        batch_size = features.shape[0]

        # s_hidden: (num_layers, batch_size, hidden_size)
        s_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device)

        # decoder_input: (batch_size, hidden_size), always start with <sos> token
        decoder_input = self._tokenized_to_hidden(self.sos_tokenized.unsqueeze(0).expand(batch_size, -1))\
            .to(features.device)

        # outputs: (batch_size, max_len, vocab_size)
        outputs = torch.zeros(batch_size, out_seq_len, self.vocab_size, device=features.device)

        for t in range(out_seq_len):
            y_t, s_hidden, _ = self._forward(decoder_input, s_hidden, features)
            outputs[:, t, :] = y_t
            decoder_input = self._tokenized_to_hidden(captions[t] if teacher_forcing else y_t.argmax(dim=-1))
        return outputs

    def forward(self, features: torch.Tensor,
                captions: torch.Tensor = None, teacher_forcing: bool = False,) -> Dict[str, torch.Tensor]:
        """
        The main method of the decoder. If captions are provided, the method will run in training mode.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        NOTE it can be another type of features, but it must be meaningful for the attention layer.
        :param captions: captions to train on (batch_size, seq_len).
        :param teacher_forcing: if True, train with teacher forcing.
        :return: output predictions (batch_size, seq_len - 1, vocab_size) if captions are provided,
        else (batch_size, any_seq_len, vocab_size)
        """
        if captions is not None:
            outputs = self._forward_training(features, captions, teacher_forcing)  # (batch_size, seq_len, vocab_size)
        else:
            # in that case sequence length is not known
            outputs = self._forward_inference(features)  # (batch_size, any_seq_len <= self.max_len, vocab_size)

        return {LOGITS_KEY: outputs}
