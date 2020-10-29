import math
import torch
import torch.nn as nn

""" MASKS UTILS """
def _generate_subsequent_mask(src_sz, tgt_sz):
    mask = (torch.triu(torch.ones(src_sz, tgt_sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _generate_square_subsequent_mask(sz):
    return _generate_subsequent_mask(sz, sz)


""" EMBEDDING UTILS """
def Embedding(num_embeddings, embedding_dim, padding_idx):
    """ Generates embeddings for tokens in vocabulary
        Weights initialized with mean=0 and std=sqrt(embedding_dim)"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


""" POSITIONAL ENCODING UTILS """
class PositionalEncoding(nn.Module):
    """ Adds positional encoding to sequences """
    def __init__(self, embedding_dim, dropout=0.1, max_seq_len=100):
        """ Initializes a seq_len x 1 x embedding_dim positional encoding matrix"""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ Adds positional encoding to the input.
            Input of dimensions (seq_len x batch_sz x embedding_dim).
            Adds positional encoding matrix (seq_len x 1 x embedding_dim) to every individual example in batch """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)