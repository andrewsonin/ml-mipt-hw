import numpy.random as rnd
import torch
from torch import nn

USE_BIDIR_ENCODER = False


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=USE_BIDIR_ENCODER)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.re_embedding = nn.Linear(enc_hid_dim * (2 if USE_BIDIR_ENCODER else 1), dec_hid_dim)
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


class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim + (2 if USE_BIDIR_ENCODER else 1) * enc_hid_dim, dec_hid_dim, dropout=dropout)
        self.out = nn.Linear(dec_hid_dim + (2 if USE_BIDIR_ENCODER else 1) * enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        scores = self.attention(hidden, encoder_outputs)
        # encoder_outputs = [query, batch, enc_hid_dim]
        # scores          = [query, batch]
        w_t = torch.einsum('qbe,qb->be', encoder_outputs, scores)
        # w_t             = [batch, query]
        w_t = w_t.unsqueeze(0)

        input = self.embedding(input.unsqueeze(0))

        catted = torch.cat((input, w_t), dim=-1)
        #print(catted.shape, hidden.shape, cell.shape)
        output, (hidden, cell) = self.rnn(
            catted,
            (hidden, cell)
        )
        output = self.out(torch.cat((input, hidden, w_t), dim=-1)).squeeze(0)
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
        if USE_BIDIR_ENCODER:
            hidden = hidden.mean(0).unsqueeze(0)
            cell = cell.mean(0).unsqueeze(0)

        # first input to the decoder is the <sos> tokens
        input = trg[0]

        use_teacher_forcing = rnd.sample(max_len - 1) < teacher_forcing_ratio
        for t, teacher_forcing in enumerate(use_teacher_forcing, 1):
            #print(enc_states.shape, hidden.shape, cell.shape)
            output, hidden, cell = self.decoder(input, hidden, cell, enc_states)
            outputs[t] = output
            input = (trg[t] if teacher_forcing else output.max(1)[1])

        return outputs
