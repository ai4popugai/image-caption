import time
from typing import Optional

import torch
from torch import nn

from nn_models.attention import MultiHeadAttention
from nn_models.positional_encoding import PositionalEncoding
from nn_models.utils import PositionWiseFeedForward


class TransformerEncoderUnit(nn.Module):
    def __init__(self, hidden_size: int, d_ff: int, num_heads: int, dropout: float):
        """
        That Unit could be use for Transformer Encoder construction for images or text encoding.

        :param hidden_size: hidden size of the model.
        :param d_ff: hidden dim for feed- forward block.
        :param num_heads: num multi-head attention blocks.
        :param dropout: dropout amount (float).
        """
        if hidden_size % num_heads != 0:
            raise RuntimeError('hidden_size must be divisible by num_heads')
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feed_forward = PositionWiseFeedForward(self.hidden_size, d_ff)
        self.self_attention = MultiHeadAttention(query_hidden_size=self.hidden_size,
                                                 keys_hidden_size=self.hidden_size,
                                                 values_hidden_size=self.hidden_size,
                                                 out_hidden_size=self.hidden_size,
                                                 hidden_size=self.hidden_size,
                                                 num_heads=self.num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_0 = nn.LayerNorm(self.hidden_size)
        self.layer_norm_1 = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        The main class method

        :param x: input source tensor with shape (batch_size, seq_len, hidden_size).
        :param mask: mask for self- attention with shape (batch_size, seq_len, seq_len).
        :return: encoded tensor with shape (batch_size, seq_len, hidden_size).
        """
        if x.shape[-1] != self.hidden_size:
            raise RuntimeError(f'Could not encode tensor with depth {x.shape[-1]}, '
                               f'only depth={self.hidden_size} allowed')
        self_attention_out, _ = self.self_attention(query=x, keys=x, values=x, mask=mask)
        x = self.layer_norm_0(x + self.dropout(self_attention_out))
        feed_forward_out = self.feed_forward(x)
        x = self.layer_norm_1(x + self.dropout(feed_forward_out))
        return x


class TransformerDecoderUnit(nn.Module):
    def __init__(self, hidden_size: int, keys_hidden_size: int,
                 d_ff: int, num_heads: int, dropout: float):
        """
        That Unit could be use for Transformer Decoder construction for images or text decoding.

        :param hidden_size: hidden size of the model.
        :param keys_hidden_size: hidden_size of keys in decoder's cross attention.
        :param d_ff: hidden dim for feed- forward block.
        :param num_heads: num multi-head attention blocks.
        :param dropout: dropout amount (float).
        """
        if hidden_size % num_heads != 0:
            raise RuntimeError('hidden_size must be divisible by num_heads')
        super().__init__()
        self.hidden_size = hidden_size
        self.keys_hidden_size = keys_hidden_size
        self.num_heads = num_heads
        self.feed_forward = PositionWiseFeedForward(self.hidden_size, d_ff)
        self.self_attention = MultiHeadAttention(query_hidden_size=self.hidden_size,
                                                 keys_hidden_size=self.hidden_size,
                                                 values_hidden_size=self.hidden_size,
                                                 out_hidden_size=self.hidden_size,
                                                 hidden_size=self.hidden_size,
                                                 num_heads=self.num_heads)
        self.cross_attention = MultiHeadAttention(query_hidden_size=self.hidden_size,
                                                  keys_hidden_size=self.keys_hidden_size,
                                                  values_hidden_size=self.keys_hidden_size,
                                                  out_hidden_size=self.hidden_size,
                                                  hidden_size=self.hidden_size,
                                                  num_heads=self.num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_0 = nn.LayerNorm(self.hidden_size)
        self.layer_norm_1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_size)

    @staticmethod
    def get_self_attention_mask(x: torch.Tensor):
        bs, trg_seq_len, _ = x.shape
        mask = torch.tril(torch.ones((trg_seq_len, trg_seq_len), dtype=torch.bool))
        return mask.unsqueeze(0).expand(bs, -1, -1)

    def forward(self, x: torch.Tensor, keys: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_cross: Optional[torch.Tensor] = None):
        """
        The main class method

        :param x: input target tensor with shape (batch_size, trg_seq_len, hidden_size)
        :param keys: tensor with shape (batch_size, keys_seq_len, keys_hidden_size) for cross attention,
        usually encoder output.
        :param mask: mask for self- attention (batch_size, trg_seq_len, trg_seq_len).
        :param mask_cross: mask for cross- attention (batch_size, trg_seq_len, keys_seq_len)
        :return: decoded tensor with shape (batch_size, trg_seg_len, hidden_size)
        """
        if mask is None:
            mask = self.get_self_attention_mask(x)
        if keys.shape[-1] != self.keys_hidden_size:
            raise RuntimeError(f'Cross attention keys hidden size '
                               f'{keys.shape[-1]} != {self.keys_hidden_size} missmatch')
        self_attention_out, _ = self.self_attention(query=x, keys=x, values=x, mask=mask)
        x = self.layer_norm_0(x + self.dropout(self_attention_out))
        cross_attention_out, _ = self.cross_attention(query=x,
                                                      keys=keys, values=keys,
                                                      mask=mask_cross)
        x = self.layer_norm_1(x + self.dropout(cross_attention_out))
        feed_forward_out = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(feed_forward_out))
        return x


class TransformerTextEncoder(nn.Module):
    def __init__(self, src_vocab_size: int,
                 num_layers: int, hidden_size: int, d_ff: int, num_heads: int, dropout: float,
                 max_seq_len: int = 200):
        super().__init__()
        self.embedder = nn.Embedding(src_vocab_size, hidden_size)
        self.encoder = nn.ModuleList([TransformerEncoderUnit(hidden_size=hidden_size,
                                                             d_ff=d_ff,
                                                             num_heads=num_heads,
                                                             dropout=dropout) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(hidden_size=hidden_size, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Encoder gets as input Tensor with shape (batch_size, seq_len).

        :param x: source tensor with shape (batch_size, seq_len).
        :param mask: self- attention mask with shape (batch_size, seq_len, seq_len)
        :return: tensor with shape (batch_size, seq_len, hidden_size)
        """
        x = self.embedder(x)
        x = self.positional_encoding(x)
        for layer in self.encoder:
            x = layer(x, mask=mask)
        return x


class TransformerTextDecoder(nn.Module):
    def __init__(self, trg_vocab_size: int, num_layers: int,
                 hidden_size: int, keys_hidden_size: int,
                 d_ff: int, num_heads: int, dropout: float,
                 max_seq_len: int = 200):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedder = nn.Embedding(trg_vocab_size, hidden_size)
        self.decoder = nn.ModuleList([TransformerDecoderUnit(hidden_size=hidden_size,
                                                             keys_hidden_size=keys_hidden_size,
                                                             d_ff=d_ff,
                                                             num_heads=num_heads,
                                                             dropout=dropout) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(hidden_size=hidden_size, max_seq_len=self.max_seq_len)
        self.fc = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, x: torch.Tensor, keys: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_cross: Optional[torch.Tensor] = None):
        """
        Decoder gets as input Tensor with shape (batch_size, trg_seq_len).

        :param x: input target tensor with shape (batch_size, trg_seq_len)
        :param keys: tensor with shape (batch_size, keys_seq_len, keys_hidden_size) for cross attention,
        usually encoder output.
        :param mask: mask for self- attention (batch_size, trg_seq_len, trg_seq_len).
        :param mask_cross: mask for cross- attention (batch_size, trg_seq_len, keys_seq_len)
        :return: decoded tensor with shape (batch_size, trg_seg_len, trg_vocab_size)
        """
        x = self.embedder(x)
        x = self.positional_encoding(x)
        for layer in self.decoder:
            x = layer(x, keys, mask=mask, mask_cross=mask_cross)
        x = self.fc(x)
        return x


class TransformerTextDecoderInference(nn.Module):
    def __init__(self, transformer_decoder: TransformerTextDecoder,
                 sos_token: torch.Tensor,
                 eos_token: torch.Tensor):
        super().__init__()
        self.transformer_decoder = transformer_decoder
        self.sos_token = sos_token
        self.eos_token = eos_token

    def forward(self, keys: torch.Tensor, ):
        """
        Class for transformer decoder inference, x argument is absent because we don't have target sequence and start
        generation from sos token.

        :param keys: tensor with shape (1, keys_seq_len, keys_hidden_size) for cross attention,
        usually encoder output.
        :return: decoded tensor with shape (1, some_seg_len, trg_vocab_size).
        NOTE! outputs will be without SOS token.
        """
        input_tokens = self.sos_token.to(keys.device).unsqueeze(0).unsqueeze(0)  # (1, 1)
        outputs = torch.empty(0)

        with torch.no_grad():
            for _ in range(self.transformer_decoder.max_seq_len):
                preds = self.transformer_decoder(x=input_tokens, keys=keys)
                # (1, some_seq_len, trg_vocab_size)
                pred = preds.select(1, -1).unsqueeze(1)  # (1, 1, trg_vocab_size)

                outputs = pred if outputs.numel() == 0 else torch.cat((outputs, pred), dim=1)
                # (1, some_seq_len, trg_vocab_size)
                pred_token = pred.argmax(dim=-1)  # (1, 1)

                if (pred_token.squeeze(0).squeeze(0) == self.eos_token).all().item():
                    return outputs
                input_tokens = torch.cat((input_tokens, pred_token), dim=1)
                # (1, some_seq_lem)
            
            return outputs


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int,
                 num_layers: int, hidden_size: int, d_ff: int, num_heads: int, dropout: float,
                 max_seq_len: int = 200):
        super().__init__()
        self.encoder = TransformerTextEncoder(src_vocab_size=src_vocab_size, num_layers=num_layers,
                                              hidden_size=hidden_size, d_ff=d_ff, num_heads=num_heads,
                                              dropout=dropout, max_seq_len=max_seq_len)
        self.decoder = TransformerTextDecoder(trg_vocab_size=trg_vocab_size, num_layers=num_layers,
                                              hidden_size=hidden_size, keys_hidden_size=hidden_size,
                                              d_ff=d_ff, num_heads=num_heads,
                                              dropout=dropout, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_cross: Optional[torch.Tensor] = None
                ):
        keys = self.encoder(x)
        return self.decoder(y, keys, mask=mask, mask_cross=mask_cross)


if __name__ == "__main__":
    # Example usage
    hs = 512
    keys_hs = 1024
    dff = 2048
    nh = 4

    # Check TransformerEncoderUnit and TransformerDecoderUnit
    print('Running TransformerEncoderUnit and TransformerDecoderUnit')
    batch_size = 16
    seq_length = 17
    keys_seq_length = 20

    # encoder
    encoder = TransformerEncoderUnit(hidden_size=hs, d_ff=dff, num_heads=nh, dropout=0.2)
    enc_inp_tensor = torch.randn(batch_size, seq_length, hs)
    print("Encoder Input Tensor Shape:", enc_inp_tensor.shape)
    start_time = time.perf_counter()
    enc_out_tensor = encoder(enc_inp_tensor)
    print("Encoder Output Tensor Shape:", enc_out_tensor.shape)
    print(f'Passing through encoder time: {time.perf_counter() - start_time}')
    print('\n')

    # decoder
    decoder = TransformerDecoderUnit(hidden_size=hs, keys_hidden_size=keys_hs,
                                     d_ff=dff, num_heads=nh, dropout=0.2)
    dec_inp_tensor = torch.randn(batch_size, seq_length, hs)
    k = torch.randn(batch_size, keys_seq_length, keys_hs)
    print("Decoder Input Tensor Shape:", dec_inp_tensor.shape)
    print("Decoder Keys Shape:", k.shape)
    start_time = time.perf_counter()
    dec_out_tensor = decoder(x=dec_inp_tensor, keys=k)
    print("Decoder Output Tensor Shape:", dec_out_tensor.shape)
    print(f'Passing through decoder time: {time.perf_counter() - start_time}')
    print('\n')

    # Check TransformerTextEncoder and TransformerTextDecoder
    src_vs = 110000
    trg_vs = 120000
    nl = 4
    max_seq_length = 100

    # encoder
    encoder = TransformerTextEncoder(src_vocab_size=src_vs, num_layers=nl, max_seq_len=max_seq_length,
                                     hidden_size=hs, d_ff=dff, num_heads=nh, dropout=0.2)
    enc_inp_tensor = torch.randint(low=0, high=max_seq_length, size=(batch_size, seq_length), dtype=torch.int32)
    print("Encoder Input Tensor Shape:", enc_inp_tensor.shape)
    start_time = time.perf_counter()
    enc_out_tensor = encoder(enc_inp_tensor)
    print("Encoder Output Tensor Shape:", enc_out_tensor.shape)
    print(f'Passing through encoder time: {time.perf_counter() - start_time}')
    print('\n')

    # decoder
    decoder = TransformerTextDecoder(trg_vocab_size=trg_vs, num_layers=nl, max_seq_len=max_seq_length,
                                     hidden_size=hs, keys_hidden_size=keys_hs,
                                     d_ff=dff, num_heads=nh, dropout=0.2)
    dec_inp_tensor = torch.randint(low=0, high=max_seq_length, size=(batch_size, seq_length), dtype=torch.int32)
    k = torch.randn(batch_size, keys_seq_length, keys_hs)
    print("Decoder Input Tensor Shape:", dec_inp_tensor.shape)
    print("Decoder Keys Shape:", k.shape)
    start_time = time.perf_counter()
    dec_out_tensor = decoder(x=dec_inp_tensor, keys=k)
    print("Decoder Output Tensor Shape:", dec_out_tensor.shape)
    print(f'Passing through decoder time: {time.perf_counter() - start_time}')
    print('\n')

    # decoder inference
    sos = torch.tensor(0)
    eos = torch.tensor(1)
    batch_size = 1
    decoder_inference = TransformerTextDecoderInference(decoder, sos_token=sos, eos_token=eos)
    k = torch.randn(batch_size, keys_seq_length, keys_hs)
    print("Decoder Inference Keys Shape:", k.shape)
    start_time = time.perf_counter()
    dec_out_tensor = decoder_inference(keys=k)
    print("Decoder Inference Output Tensor Shape:", dec_out_tensor.shape)
    print(f'Passing through inference decoder time: {time.perf_counter() - start_time}')
    print('\n')

    # transformer
    transformer = Transformer(src_vocab_size=src_vs, trg_vocab_size=trg_vs, num_layers=nl, max_seq_len=max_seq_length,
                              hidden_size=hs, d_ff=dff, num_heads=nh, dropout=0.2)
    batch_size = 16
    src_seq_length = 20
    trg_seq_length = 17
    src = torch.randint(low=0, high=max_seq_length, size=(batch_size, src_seq_length), dtype=torch.int32)
    dst = torch.randint(low=0, high=max_seq_length, size=(batch_size, trg_seq_length), dtype=torch.int32)
    print("Transformer Source Tensor Shape:", src.shape)
    print("Transformer Target Tensor Shape:", dst.shape)
    start_time = time.perf_counter()
    transformer_out = transformer(src, dst)
    print("Transformer Output Tensor Shape:", transformer_out.shape)
    print(f'Passing through transformer time: {time.perf_counter() - start_time}')
    print('\n')
