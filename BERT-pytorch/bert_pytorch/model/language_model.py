import torch
import torch.nn as nn

from .bert import BERT
from .bert import ALBERT
from .utils import GELU
from .utils import LayerNorm

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class ALBERTLM(nn.Module):
    """
    ALBERT Language Model
    Sentence Order Prediction Model + Masked Language Model
    """

    def __init__(self, bert: ALBERT, embed_size, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.sop = SentenceOrderPrediction(self.bert.hidden)
        self.mlm = MaskedLanguageModel(self.bert.hidden, embed_size, vocab_size, embedding=self.bert.embedding.token)

    def forward(self, x, segment_label, masked_pos):
        x = self.bert(x, segment_label)
        return self.sop(x), self.mlm(x, masked_pos)        
        


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        self.fc   = nn.Linear(hidden, hidden)
        self.act  = nn.Tanh()
        self.clsf = nn.Linear(hidden, 2)      

    def forward(self, x):
        x = self.fc(x[:,0])
        x = self.act(x)
        x = self.clsf(x)
        return x



class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, embed_size, vocab_size, embedding):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """

        # ALBERT에서는 embedding size가 hidden size보다 작다. albert의 output은 B * seq_len * hidden_size인데, embedding matrix를 활용하기 위해서는 hidden -> embed_size로 변환이 필요하다.
        super().__init__()

        self.fc    = nn.Linear(hidden, embed_size)
        self.act   = GELU()        
        self.norm  = LayerNorm(embed_size)
        self.clsf  = nn.Linear(embed_size, vocab_size, bias=False)  # why??
        self.clsf_bias = nn.Parameter(torch.zeros(vocab_size))      # why ??

        self.clsf.weight.data = embedding.weight.data        

    def forward(self, x, masked_pos):
        masked_pos = masked_pos[:, :, None].expand(-1, -1, x.size(-1))
        h_masked = torch.gather(x, 1, masked_pos)            # B * seq_len * hidden_size -> B * #randomized token * hidden_size

        h_masked = self.fc(h_masked)      # -> B * #randomized token * embed_size
        h_masked = self.act(h_masked)
        h_masked = self.norm(h_masked)
        h_masked = self.clsf(h_masked) + self.clsf_bias   # -> B * #randomized token * vocab_size
        return h_masked


class SentenceOrderPrediction(nn.Module):
    """
    2-class classification model : ordered, reversed
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.fc   = nn.Linear(hidden, hidden)
        self.act  = nn.Tanh()
        self.clsf = nn.Linear(hidden, 2)      

    def forward(self, x):
        x = self.fc(x[:,0])
        x = self.act(x)
        x = self.clsf(x)
        return x

        '''
        x = self.fc(x)
        x = self.act(x)
        x = self.clsf(x)
        x = torch.sum(x, dim=1)
        return x
        '''
