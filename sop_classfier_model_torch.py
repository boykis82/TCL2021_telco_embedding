import torch
import torch.nn as nn
from torch.utils.data import Dataset

import textlib as tl

class SOPClassifier(nn.Module):
    def __init__(self, bert, num_class, dropout):
        super().__init__()
        self.bert = bert
        self.decode1 = nn.Linear(self.bert.hidden, self.bert.hidden)
        self.dropout = nn.Dropout(dropout)
        self.decode2 = nn.Linear(self.bert.hidden, num_class)
        
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        h = self.decode1(x[:, 0])
        h = self.dropout(h)
        out = self.decode2(h)
        return out    


class SOPDataset(Dataset):
    def __init__(self, X, y, vocab, seq_len):
        self.vocab = vocab
        self.seq_len = seq_len
        self.X = []
        if y is not None:
            self.y = torch.LongTensor(y)
        else:
            self.y = None
        
        print(f'data loading started! size = {len(X)}')
        for i, text in enumerate(X):
            try:
                # 클렌징
                cleansed_text = tl.clean_text(text)
            except TypeError:
                print(f'      {i+1} 번째 데이터에 문제가 있어 skip!')
                continue

            # 문장으로 분리하여 배열로 리턴
            sentences = tl.segment_sentences(cleansed_text)
            # 문장 배열을 입릭으로 받아 형태소로 쪼갠 뒤, 다시 하나의 문자열로 변환하여 저장
            corpora = ' '.join(tl.get_corpora(sentences)).split(' ')
            self.X.append(corpora)   
            
            if i%1000 == 0:
                print(f'{i}th data loading completed!')
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, item):
        x = self.X[item]        
        x = x[:(self.seq_len-2)]

        tokens = ['<sos>'] + x + ['<eos>']
        segment_ids = [0] * len(tokens)
        
        input_ids = self.vocab.to_seq(tokens)
        
        n_pad = self.seq_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        
        if self.y is None:
            output = {'input_ids': input_ids,
                    'segment_ids': segment_ids}            
        else:
            output = {'input_ids': input_ids,
                    'segment_ids': segment_ids,
                    'label': self.y[item]}

        return {key: torch.tensor(value) for key, value in output.items()}        