from os.path import dirname, join, realpath

from fastapi import FastAPI

from bert_pytorch.model import ALBERT
from bert_pytorch.dataset import WordVocab

import torch
import torch.nn as nn

from sop_classfier_model_torch import SOPClassifier, SOPDataset

# sop class 개수
num_class = 37

# 최대 문장 길이 (한 문장에 포함될 최대 단어 개수. 조사, 동사 같이 예측에 도움안되는 품사들은 모두 잘려나가므로 생각보다 많이 담을 수 있다.)
seq_len = 64

# vocabulary 경로
vocab_path = '../TCL2021_Telco_Embedding_Dataset/corpora/telco_vocab.dat'
vocab = WordVocab.load_vocab(vocab_path)

# 분류기 모듈 경로(pytorch)
clf_model_path = 'sop_clf_bert_model_checkpoint.pt'
# ALBERT 모델
bert = ALBERT(vocab_size=len(vocab), embed_size=128, hidden=256, n_layers=8, attn_heads=8, seq_len=64)
# SOP 분류기
clf = SOPClassifier(bert, num_class, 0.0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 load후 평가 모드로 전환
clf.load_state_dict(torch.load(clf_model_path))
clf.to(device)
clf.eval()

app = FastAPI(
    title="SKT/B SOP Classfier",
    description="SKT/B SOP Classfier",
    version="0.5"
)            

label_map = {   
    0:  'MPAMS',
    1:  'MVNO',
    2:  'SKT eService-Tsales',
    3:  'SWING CTC',
    4:  'SWING DBM',
    5:  'SWING NIS',
    6:  'SWING Payment',
    7:  'SWING Portal',
    8:  'SWING SSO',
    9:  'SWING 개통',
    10: 'SWING 단말',
    11: 'SWING 멤버십',
    12: 'SWING 모바일',
    13: 'SWING 미납',
    14: 'SWING 스마트플래너',
    15: 'SWING 시설',
    16: 'SWING 오더 - 무선오더',
    17: 'SWING 유선OSS 모바일',
    18: 'SWING 유선상품',
    19: 'SWING 유선오더',
    20: 'SWING 자원 - 계약서관리',
    21: 'SWING 장애',
    22: 'SWING 주소',
    23: 'SWING 청구',
    24: 'SWING 파트너관리(PRM)',
    25: 'T gate',
    26: 'UI, 프레임웍',
    27: '과금계산',
    28: '과금정보',
    29: '기타',
    30: '상품-Interface',
    31: '상품-단말기',
    32: '상품-무선',
    33: '서식지통합관리',
    34: '소액결제-ISAS',
    35: '인터페이스(SMS,MMS포함)',
    36: '판매점 SSO'
}

# uvicorn sop_clf_web_api:app --reload
@app.get('/predict')
def predict_sentiment(sentence):
    print(f'----------------------- {sentence} -------------------------')

    # SOPDataset 생성. 안에서 전처리 & tokenizing & padding 등 수행
    tensor = SOPDataset([sentence], None, vocab, seq_len)[0]

    data = {key: value.to(device) for key, value in tensor.items()}
    # 입력의 0번째 차원이 batch size이므로 [n] -> [1,n] 으로 변환
    input_ids = data["input_ids"].view(1,-1)
    segment_ids = data["segment_ids"].view(1,-1)
    # 전처리 후 vocab id로 변경된 결과
    print(f'input_ids : {input_ids}')

    # 예측
    pred_y = clf.forward(input_ids, segment_ids)[0]
    pred_y = nn.functional.softmax(pred_y, dim=-1)
    print(f'softmax : {pred_y}')

    # top 5에 해당되는 확률과 index 가져옴
    probs, indices = torch.topk(pred_y, 5)
    probs *= 100
    probs = probs.cpu().detach().numpy()
    indices = indices.cpu().detach().numpy()

    # index로 되어 있는 label을 문자열로 변환
    label = [label_map[l] for l in indices]

    # 결과 리턴
    result = [f'{l}-{round(p,2)}' for l,p in zip(label, probs)]
    print(f'result: {result}')

    return result