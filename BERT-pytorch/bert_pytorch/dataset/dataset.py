from torch.utils.data import Dataset
import tqdm
import torch
import random
from random import shuffle

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                #prob /= 0.15
                prob = random.random()

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% keep
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if t2 is None:
            return t1, self.get_random_line(), 0
        else:
            return t1, t2, 1

    def get_corpus_line(self, item):
        if self.on_memory:
            if len(self.lines[item]) == 1:
                return self.lines[item][0], None
            else:
                return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            unpacked = line[:-1].split("\t")
            if len(unpacked) == 1:
                return unpacked[0], None
            else:
                return unpacked[0]. unpacked[1]

    def get_random_line(self):
        if self.on_memory:
            line = self.lines[random.randrange(len(self.lines))]
            return line[0] if len(line) == 1 else line[1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()

        unpacked = line[:-1].split("\t")
        if len(unpacked) == 1:
            return unpacked[0]
        else:
            return unpacked[1]
        


class ALBERTDataset(Dataset):
    @staticmethod
    def create_dataset(corpus_path, vocab, seq_len, max_pred=10, mask_prob=0.15, augmentation_count=5, train_ratio: float=0.8, sample_ratio: float=1.0, encoding="utf-8"):
        data = []

        line1, line2 = None, None

        with open(corpus_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f.readlines()):
                if line == '\n':
                    line1 = None
                    line2 = None
                    continue

                if line1 is None:
                    line1 = line
                else:
                    line2 = line
                    for i in range(augmentation_count):
                        data.append( (line1[:-1], line2[:-1]) )
                    line1 = line2        

        count = len(data)
        if sample_ratio < 1.0:
            count = round(len(data) * sample_ratio)
            shuffle(data)
            data = data[:count]

        train_cnt = int(len(data) * train_ratio)

        train_dataset = ALBERTDataset(vocab, seq_len, data[:train_cnt], max_pred, mask_prob)
        if train_ratio < 1.0:
            test_dataset  = ALBERTDataset(vocab, seq_len, data[train_cnt:], max_pred, mask_prob)
            return train_dataset, test_dataset
        else:
            return train_dataset, None

        


    def __init__(self, vocab, seq_len, data, max_pred=10, mask_prob=0.15):
        self.vocab = vocab
        self.seq_len = seq_len

        self.data = data
        self.max_pred = max_pred
        self.mask_prob = mask_prob


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        t1, t2, ordered_label = self.mix_sent_order(item)

        '''
        input_ids : t1 + t2 의 vocab ids
        segment_ids : sent1 or sent2?
        input_mask: padding부분은 죽이기 위한 mask
        masked_ids: t1 + t2 에서 masking하여 변환된 ids
        masked_pos: t1 + t2 에서 masking된 index
        masked_weights: padding부분은 죽이기 위한 mask
        '''
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights = self.mask_tokens(t1, t2)

        output = {"input_ids"     : input_ids,
                  "segment_ids"   : segment_ids,
                  "input_mask"    : input_mask,
                  "masked_ids"    : masked_ids,
                  "masked_pos"    : masked_pos,
                  "masked_weights": masked_weights,
                  "ordered"       : ordered_label}

        return {key: torch.tensor(value) for key, value in output.items()}


    def mask_tokens(self, t1, t2):
        # special token 넣을 자리 확보
        t1, t2 = self.truncate_tokens_pair(t1, t2, self.seq_len - 3)

        # <sos> t1 <eos> t2 < eos>
        tokens = ['<sos>'] + t1 + ['<eos>'] + t2 + ['<eos>']
        segment_ids= [0] * (len(t1) + 2) + [1] * (len(t2) + 1)
        input_mask = [1] * len(tokens)

        # For MLM
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens) if token != '<sos>' and token != '<eos>']        
        shuffle(cand_pos)

        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            prob = random.random()
            if prob < 0.8: # 80%
                tokens[pos] = '<mask>'
            elif prob < 0.9: # 10%
                rnd = random.randint(5, len(self.vocab)-1)    # special token 제외
                tokens[pos] = self.vocab.itos[rnd]
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.vocab.to_seq(tokens)
        masked_ids = self.vocab.to_seq(masked_tokens)

        # Zero Padding
        n_pad = self.seq_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights)


    def truncate_tokens_pair(self, t1, t2, max_len):
        while True:
            if len(t1) + len(t2) <= max_len:
                break
            if len(t1) > len(t2):
                t1.pop()
            else:
                t2.pop()    
        return t1, t2               


    def mix_sent_order(self, ndx):
        t1, t2 = self.data[ndx][0], self.data[ndx][1]

        # output_text, label(reversed:0, ordered:1)
        if random.random() > 0.5:
            return t1.split(), t2.split(), 1
        else:
            return t2.split(), t1.split(), 0            
