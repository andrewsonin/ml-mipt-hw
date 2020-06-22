import math

import numpy.random as rnd
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.re_embedding = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = self.re_embedding(encoder_outputs)
        # [query, batch, enc_hid_dim] -> [query, batch, dec_hid_dim]

        # hidden: [batch, dec_hid_dim]
        # encoder_outputs_{q,b,d} * hidden_{d,b} = scores_{q,b}
        scores = torch.einsum('qbd,bd->qb', encoder_outputs, hidden[0])
        # Calculates pairwise dot product of `hidden` and each query element for batch
        scores = self.softmax(scores)
        return scores  # [query, batch]


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = hid_dim
        self.dec_hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.attention = Attention(self.enc_hid_dim, self.dec_hid_dim)

        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)

        self.rnn = nn.LSTM(
            self.emb_dim + self.enc_hid_dim,
            self.dec_hid_dim,
            self.n_layers,
            dropout=dropout
        )
        self.out = nn.Linear(self.dec_hid_dim * self.n_layers + self.enc_hid_dim + self.emb_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        scores = self.attention(hidden, encoder_outputs)
        # encoder_outputs = [query, batch, enc_hid_dim]
        # scores          = [query, batch]
        w_t = torch.einsum('qbe,qb->be', encoder_outputs, scores)
        # w_t             = [batch, query]
        w_t = w_t.unsqueeze(0)

        input = self.embedding(input.unsqueeze(0))

        output, (hidden, cell) = self.rnn(
            torch.cat((input, w_t), dim=-1),
            (hidden, cell)
        )
        w_t = w_t.squeeze(0)
        input = input.squeeze(0)
        hidden_flattened = hidden.permute(1, 2, 0).reshape(hidden.shape[1], -1)
        # [n_layers, n_batches, hid_dims] -> [n_batches, hid_dims * n_layers]
        # Flattens along layers and passes hidden state of each layer to self.out
        output = self.out(
            torch.cat(
                (input, hidden_flattened, w_t),
                dim=-1
            )
        )
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.dec_hid_dim,
            "Hidden dimensions of the encoder and decoder must be equal!"
        )
        assert (
            encoder.n_layers == decoder.n_layers,
            "Encoder and decoder must have equal number of layers!"
        )

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        max_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size, device=self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0]

        use_teacher_forcing = rnd.sample(max_len - 1) < teacher_forcing_ratio
        for t, teacher_forcing in enumerate(use_teacher_forcing, 1):
            # print(enc_states.shape, hidden.shape, cell.shape)
            output, hidden, cell = self.decoder(input, hidden, cell, enc_states)
            outputs[t] = output
            input = trg[t] if teacher_forcing else output.max(1)[1]

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)

        pe = torch.empty(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            (-math.log(10000) / d_model) * torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        )
        tensor_dot = position * div_term
        pe[:, 0::2] = torch.sin(tensor_dot)
        pe[:, 1::2] = torch.cos(tensor_dot)

        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        x += self.pe[:x.shape[0]]
        return x


class Trans(nn.Module):
    def __init__(self, n_hidden, input_dim, output_dim, pe, device):
        super().__init__()

        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.pe = pe
        #self.pe = PositionalEncoding(n_hidden, device=device)
        self.inp_embedding = nn.Embedding(self.input_dim, self.n_hidden)
        self.out_embedding = nn.Embedding(self.output_dim, self.n_hidden)
        self.transformer = nn.Transformer(
            self.n_hidden,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        self.out = nn.Linear(self.n_hidden, self.output_dim)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def generate_square_subsequent_mask(size, device=None) -> torch.tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).

            tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
                    [0.,   0., -inf,  ..., -inf, -inf, -inf],
                    [0.,   0.,   0.,  ..., -inf, -inf, -inf],
                    ...,
                    [0.,   0.,   0.,  ...,   0., -inf, -inf],
                    [0.,   0.,   0.,  ...,   0.,   0., -inf],
                    [0.,   0.,   0.,  ...,   0.,   0.,   0.]])
         """
        return torch.tensor(float('-inf'), device=device).repeat(size, size).triu(1)

    def forward(self, src, trg):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        max_len, batch_size = trg.shape
        # print(trg.shape)
        # tensor to store decoder outputs

        src = self.inp_embedding(src)
        trg = self.out_embedding(trg)
        output = self.transformer(
            self.pe(src), self.pe(trg),
            tgt_mask=self.generate_square_subsequent_mask(max_len, device=self.device)
        )
        return self.softmax(self.out(output))
