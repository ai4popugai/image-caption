from typing import Dict

import torch
from torch import nn

from datasets import LOGITS_KEY
from nn_models.attention import BahdanauAttention
from train import Trainer


class AttentionDecoder(nn.Module):
    def __init__(self, trg_vocab_size: int, hidden_size: int, keys_hidden_size: int,
                 sos_token: torch.Tensor, eos_token: torch.Tensor,
                 num_layers=4, max_len=15, teacher_forcing: bool = False,):
        super().__init__()
        # attention mechanism for feature maps and hidden state
        self.attention = BahdanauAttention(query_hidden_size=hidden_size, keys_hidden_size=keys_hidden_size,
                                           values_hidden_size=keys_hidden_size,
                                           hidden_size=hidden_size, out_hidden_size=hidden_size)
        self.teacher_forcing = teacher_forcing

        self.hidden_size = hidden_size
        self.trg_vocab_size = trg_vocab_size

        # embedding layer to convert words indices to embeddings
        self.embedding = nn.Embedding(trg_vocab_size, hidden_size)

        # rnn cell
        self.num_layers = num_layers

        self.rnn = nn.GRU(hidden_size + hidden_size, hidden_size, num_layers=self.num_layers, batch_first=True)

        # linear layer to make final predictions
        self.fc_2 = nn.Linear(hidden_size, trg_vocab_size)

        self.max_len = max_len
        self.sos_token = sos_token
        self.eos_token = eos_token

    def _token_to_hidden(self, tokenized_word: torch.Tensor) -> torch.Tensor:
        """
        Convert tokenized word firstly to embeddings and then to s_hidden.

        :param tokenized_word: (batch_size,) - tokenized words.
        :return: (batch_size, hidden_size) - vector that can be treated as s_hidden.
        """
        embeddings = self.embedding(tokenized_word)  # embeddings: (batch_size, hidden_size)
        return embeddings

    def _forward(self, decoder_input: torch.Tensor, s_hidden: torch.Tensor, keys: torch.Tensor,) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        One step forward of the decoder, given the hidden state s_hidden and the keys (feature maps).
        s_hidden is the hidden state of the previous time step.

        :param decoder_input: (batch_size, hidden_size) tensor from input ground true y_i-1 or previous predicted word.
        :param s_hidden: hidden state of the previous time step (num_layers, batch_size, hidden_size).
        :param keys: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :return: decoded word (batch_size, trg_vocab_size), hidden state (batch_size, hidden_size) and attention weights.
        """
        # I get hidden state (batch_size, 1, hidden_size) from the last layer of GRU to the attention layer
        context, weights = self.attention(query=s_hidden[-1].unsqueeze(1), keys=keys, values=keys)
        # context: (batch_size, 1, hidden_size)
        # weights: (batch_size, 1, seq_len)

        # rnn_input: (batch_size, 1, hidden_size + hidden_size)
        rnn_input = torch.cat([decoder_input.unsqueeze(1), context], dim=-1)

        # GRU takes rnn_input: (batch_size, 1, 2 * hidden_size
        # and s_hidden: (num_layers, batch_size, hidden_size)
        y_t, s_hidden = self.rnn(rnn_input, s_hidden)  # output: (batch_size, 1, hidden_size)
        # s_hidden: (num_layers, batch_size, hidden_size)

        y_t = y_t.squeeze(1)  # output: (batch_size, hidden_size)
        y_t = self.fc_2(y_t)  # output: (batch_size, trg_vocab_size)
        return y_t, s_hidden, weights

    def _forward_inference(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in inference mode. Captions are not provided.
        Decode the sequence step by step, until the EOS token is predicted or the max_len is reached.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :return: (batch_size, any_seq_len, trg_vocab_size) - predictions for each token in the sequence.
        """

        batch_size = features.shape[0]
        assert batch_size == 1, "In inference mode batch size must be 1."
        # s_hidden: (num_layers, batch_size, hidden_size)
        s_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device)

        # decoder_input: (batch_size, hidden_size), always start with SOS token
        decoder_input = self._token_to_hidden(self.sos_token.to(features.device)).expand(batch_size, -1)

        # outputs: (batch_size, max_len, trg_vocab_size)
        outputs = torch.zeros(batch_size, self.max_len, self.trg_vocab_size, device=features.device)

        with torch.no_grad():
            for t in range(self.max_len):
                y_t, s_hidden, _ = self._forward(decoder_input, s_hidden, features)  # y_t: (batch_size, trg_vocab_size)
                # s_hidden: (num_layers, batch_size, hidden_size)

                outputs[:, t, :] = y_t

                # get the most probable word index
                decoded_token = y_t.argmax(dim=-1)  # decoded_token: (batch_size,)

                # check if decoded_token is EOS token
                if (decoded_token == self.eos_token).all().item():
                    return outputs[:, :t + 1, :]

                # set new decoder_input
                decoder_input = self._token_to_hidden(decoded_token)

        return outputs

    def _forward_training(self, features: torch.Tensor, captions: torch.Tensor,) -> torch.Tensor:
        """
        Forward pass in training mode. Captions are provided.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        :param captions: captions to train on (batch_size, seq_len_captions).
        :return: (batch_size, seq_len_captions - 1, trg_vocab_size) - predictions for each token in the sequence
        except SOS token.
        """
        captions = captions[:, 1:]  # because we don't need to predict SOS token
        out_seq_len = captions.shape[1]
        batch_size = features.shape[0]

        # s_hidden: (num_layers, batch_size, hidden_size)
        s_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device)

        # decoder_input: (batch_size, hidden_size), always start with SOS token
        decoder_input = self._token_to_hidden(self.sos_token.to(features.device)).expand(batch_size, -1)

        # outputs: (batch_size, max_len, trg_vocab_size)
        outputs = torch.zeros(batch_size, out_seq_len, self.trg_vocab_size, device=features.device)

        for t in range(out_seq_len):
            y_t, s_hidden, _ = self._forward(decoder_input, s_hidden, features)
            outputs[:, t, :] = y_t
            decoder_input = self._token_to_hidden(captions[t] if self.teacher_forcing else y_t.argmax(dim=-1))
        return outputs

    def forward(self, features: torch.Tensor,
                captions: torch.Tensor = None,) -> Dict[str, torch.Tensor]:
        """
        The main method of the decoder. If captions are provided, the method will run in training mode.

        :param features: feature maps from the encoder (batch_size, seq_len, keys_hidden_size).
        NOTE it can be another type of features, but it must be meaningful for the attention layer.
        :param captions: captions to train on (batch_size, seq_len).
        :return: output predictions (batch_size, seq_len - 1, trg_vocab_size) if captions are provided,
        else (batch_size, any_seq_len, trg_vocab_size)
        """
        if captions is not None:
            outputs = self._forward_training(features, captions)  # (batch_size, seq_len, trg_vocab_size)
        else:
            # in that case sequence length is not known
            outputs = self._forward_inference(features)  # (batch_size, any_seq_len <= self.max_len, trg_vocab_size)

        return {LOGITS_KEY: outputs}
    

if __name__ == '__main__':
    device = Trainer.get_device()
    vs = 100000
    hs = 512
    keys_hs = 1024
    sos = torch.tensor(0)
    eos = torch.tensor(1)
    m_len = 500
    decoder = AttentionDecoder(trg_vocab_size=vs,
                               hidden_size=hs, keys_hidden_size=keys_hs,
                               sos_token=sos, eos_token=eos, max_len=m_len
                               )
    decoder.to(device)
    bs = 1
    seq_length_features = 48
    seq_length_captions = 15
    dec_inp_features = torch.randn(bs, seq_length_features, keys_hs).to(device)
    dec_inp_captions = torch.randn(bs, seq_length_captions).to(device)
    dec_out = decoder(dec_inp_features, dec_inp_captions)
    print("Decoder Input Features Shape:", dec_inp_features.shape)
    print("Decoder Output Tensor Shape:", dec_out[LOGITS_KEY].shape)
