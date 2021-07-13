import argparse

from torch.utils.data import DataLoader

from bert_pytorch.model import ALBERT
from bert_pytorch.dataset import WordVocab

import torch
import torch.nn as nn
from torch.optim import Adam

import tqdm

# numpy & pandas
import numpy as np
import pandas as pd

# scikit learn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import textlib as tl
import EarlyStopping as es

from sop_classfier_model_torch import SOPClassifier, SOPDataset

class SOPClassfierTrainer:
    def __init__(self, 
                clf: SOPClassifier, 
                train_data_loader: DataLoader, 
                valid_data_loader: DataLoader, 
                test_data_loader: DataLoader, 
                optim, 
                early_stopping: es.EarlyStopping,
                epochs: int,
                model_path: str):
        self.clf = clf
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader  = test_data_loader
        self.optim = optim
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.model_path = model_path

    def _iterate_one_epoch(self, epoch, data_loader, losses, mode: str):
        sum_loss = 0.0
        total_correct_sop = 0
        total_element_sop = 0

        mode = mode.lower()
        if mode not in ("train", "valid", "test"):
            raise ValueError("invalid mode! (train, valid, test)")

        data_iter = tqdm.tqdm(enumerate(data_loader),
                                desc="EP_%s:%d" % (mode, epoch),
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}")        
        
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            pred_y = self.clf.forward(data["input_ids"], data["segment_ids"])
            loss = self.criterion(pred_y, data["label"])

            if mode == "train": 
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            sum_loss += loss.item()

            # SOP accuracy 계산
            correct_sop = pred_y.argmax(dim=-1).eq(data["label"]).sum().item()
            total_correct_sop += correct_sop
            total_element_sop += data["label"].nelement()

            sop_acc = total_correct_sop / total_element_sop * 100

            post_fix = {
                "epoch": "[%d/%s]" % (epoch, mode),
                "iter": "[%d/%d]" % (i, len(data_loader)),
                "avg_loss": sum_loss / (i + 1),
                "sop_acc": sop_acc,
                "total_loss": loss.item()
            }

            if i % 100 == 0:
                data_iter.write(str(post_fix))

            losses.append(sum_loss / len(data_loader))

        print("EP%d_%s, avg_loss=" % (epoch, mode), sum_loss / len(data_loader), \
            "total_sop_acc=", total_correct_sop * 100.0 / total_element_sop)        

    def _train_one_epoch(self, epoch):
        self.clf.train()
        self._iterate_one_epoch(epoch, self.train_data_loader, self.train_losses, "train")

    def _validate_one_epoch(self, epoch):
        self.clf.eval()
        self._iterate_one_epoch(epoch, self.valid_data_loader, self.valid_losses, "valid")

    def train_and_validate(self):
        # GPU or CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clf.to(self.device)        

        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)
            self._validate_one_epoch(epoch)   

            self.early_stopping(self.valid_losses[-1], clf)       

            if self.early_stopping.early_stop:
                print("Early stopping!")
                break    

        self.clf.load_state_dict(torch.load(self.model_path))
        self.clf.to(self.device)

    def train_only(self):
        # GPU or CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clf.to(self.device)        

        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)

        torch.save(self.clf.state_dict(), self.model_path)

    def predict(self):
        self.clf.eval()
        self._iterate_one_epoch(0, self.test_data_loader, self.test_losses, "test")            


def train_test_valid_split(X, y, test_size=0.2, valid_size=0.2, random_state=42):
    if valid_size > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=random_state, stratify=y_train)        

        return X_train, X_test, X_valid, y_train, y_test, y_valid
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, None, X_valid, y_train, None, y_valid


# python -W ignore sop_classifier_bert.py -v ../TCL2021_Telco_Embedding_Dataset/corpora/telco_vocab.dat -m ../TCL2021_Telco_Embedding_Dataset/albert_model/albert.model_weightsonly.ep9 -c sop_clf_bert_model_checkpoint.pt -d ../TCL2021_Telco_Embedding_Dataset/dataset/sop_dataset.xlsx -t False --e 4
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument("-v", "--vocab_path", type=str, required=True)
    parser.add_argument("-m", "--bert_model_path", type=str, required=True)
    parser.add_argument("-c", "--sop_classifier_model_path", type=str, required=True)
    parser.add_argument("-d", "--sop_dataset_path", type=str, required=True)
    parser.add_argument("-t", "--train_and_validate", type=str, default="True")    
    parser.add_argument("-e", "--epochs", type=int, default=10)    
    args = parser.parse_args()

    # file paths
    vocab_path = args.vocab_path
    albert_model_path = args.bert_model_path
    clf_model_path = args.sop_classifier_model_path
    sop_dataset_path = args.sop_dataset_path
    train_and_validate = True if args.train_and_validate.lower() == 'true' else False

    # load vocabulary
    vocab = WordVocab.load_vocab(vocab_path)

    # load pretrainer albert model
    bert = ALBERT(vocab_size=len(vocab), embed_size=128, hidden=256, n_layers=8, attn_heads=8, seq_len=64)
    bert = torch.load(albert_model_path)

    # parameters
    num_class = 37
    batch_size = 64
    seq_len = 64
    epochs = 20    
    patience = 3
    dropout = 0.5

    test_size = 0.2
    val_size = 0.0

    # load sop dataset
    try:
        df = pd.read_excel(sop_dataset_path, sheet_name=0, engine='openpyxl')
    except FileNotFoundError:
        print(f'{sop_dataset_path}이 없습니다! skip!')

    # 첫 모델은 sentence와 label만 써보자
    # df_zip = df[ ['sentence', 'label'] ]
    y = df.pop('label_clean')
    X = df.pop('sentence')

    # 문자열로 되어 있는 label을 categorical value로 변환
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    #print(label_encoder.inverse_transform([i for i in range(num_class)]))

    # train / test 분리
    if train_and_validate:
        X_train, X_test, X_valid, y_train, y_test, y_valid = train_test_valid_split(X, y, test_size=test_size, valid_size=val_size, random_state=42)
    # train only 모드면 hyperparameter 튜닝 등 모두 끝나고 validation set까지 모두 함께 훈련        
    else:
        X_train, X_test, X_valid, y_train, y_test, y_valid = X, None, None, y, None, None
        
    # preparedataset         
    train_dataset = SOPDataset(X_train, y_train, vocab, seq_len)
    valid_dataset = SOPDataset(X_valid, y_valid, vocab, seq_len) if X_valid is not None else None        
    test_dataset  = SOPDataset(X_test,  y_test,  vocab, seq_len) if X_test is not None else None        

    # for weighted random sample
    '''
    class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
    class_weights = 1. / class_sample_count
    train_sampler = WeightedRandomSampler(weights=class_weights, 
                                        num_samples=len(class_weights),
                                        replacement=True)
    '''                                        

    # create dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0) if X_valid is not None else None                                
    test_data_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers=0) if X_test is not None else None        

    # sop classfier
    clf = SOPClassifier(bert, num_class, dropout)
                   
    # adam optimizer
    optim = Adam(clf.parameters(), 1e-4)             
    
    # early stopping                   
    early_stopping = es.EarlyStopping(patience, True, 0, clf_model_path)

    # trainer
    sop_clf_trainer = SOPClassfierTrainer(clf, train_data_loader, valid_data_loader, test_data_loader, optim, early_stopping, args.epochs, clf_model_path)

    # train and validate and predict
    if train_and_validate:
        sop_clf_trainer.train_and_validate()
        if X_test is not None:
            sop_clf_trainer.predict()
    else:
        sop_clf_trainer.train_only()




