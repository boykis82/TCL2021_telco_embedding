import torch
import torch.nn as nn

from ..utils import LayerNorm

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len, dropout=0.0):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token      = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position   = nn.Embedding(seq_len, embed_size, padding_idx=0)
        self.segment    = nn.Embedding(3, embed_size, padding_idx=0)    # why 3? -> 0 : padding, 1 : 1st sentence, 2 : second sentence
        
        self.norm       = LayerNorm(embed_size)

        self.dropout    = nn.Dropout(p=dropout)

        self.embed_size = embed_size
        self.seq_len    = seq_len

    def forward(self, sequence, segment_label):
        pos = torch.arange(self.seq_len, dtype=torch.long, device=sequence.device)
        pos = pos.unsqueeze(0).expand_as(sequence)

        x = self.token(sequence) + self.position(pos) + self.segment(segment_label)
        x = self.norm(x)
        return self.dropout(x)
