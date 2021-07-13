# 1. 말뭉치 수집 & 전처리
## A. Raw Data 다운로드
### ■ SOR
```
메뉴 : ITRM - 요청관리 - 요청서 조회
조회조건 :
    등록일 : 한달 단위로 입력 (ex: 2021년 2월이면 2021-02-01 ~ 2021-02-28)

조회 후 SOR폴더에 202102.xlsx로 저장  
```    

### ■ SOP
```
메뉴 : serviceflow - 장애조회
조회조건 :
    고객사 : SKT계열 
    서비스구분 : Application
    발생일자 : 한달 단위로 입력 (ex: 2021년 2월이면 2021-02-01 ~ 2021-02-28)

조회 후 SOP폴더에 202102.xlsx로 저장  
```

### ■ Jira SOR
```
메뉴 : JIRA - issue조회
조회조건 : 날짜 부분을 한달 단위로 바꿔가며 추출
    project = OP301 AND issuetype = SOR요청서  and created >= '2021-05-01' and created <= '2021-05-31' and component is not null order by created
추출컬럼 :
    요약, 생성됨, 설명, 구성요소

HTML로 download (엑셀로 받으면 인코딩 문제인지 한글이 모두 깨진다.)
조회 후 SOR_JIRA폴더에 202102.xlsx로 저장    
```    
----  
## B. Raw Data 병합 & 문장 추출
**1에서 다운로드 받은 파일이 d:/data 폴더 내 SOR, SOP, SOR_JIRA 폴더에 있다고 가정**

### 월별 SOR을 merge한 후 필요한 컬럼만 추출하여 output_path 로 지정한 파일에 쓴다. from_ym, to_ym으로 범위 지정
```
python merge_extract.py --type SOR --input_path D:/data/SOR --output_path D:/data/SOR/sor_merged.xlsx --from_ym 202101 --to_ym 202102
```
### 월별 SOP를 merge한 후 필요한 컬럼만 추출하여 output_path 로 지정한 파일에 쓴다. from_ym, to_ym으로 범위 지정
```
python merge_extract.py --type SOP --input_path D:/data/SOP --output_path D:/data/SOP/sop_merged.xlsx --from_ym 202101 --to_ym 202102
```  
### JIRA는 HTML포맷으로 다운받았기 때문에 엑셀로 변환하는 전처리를 먼저 수행해야 함
```
python conv_jira.py --input_path D:/data/SOR_JIRA --output_path D:/data/SOR_JIRA_AF --from_ym 202006 --to_ym 202105
```
### 월별 Jira SOR를 merge한 후 필요한 컬럼만 추출하여 output_path 로 지정한 파일에 쓴다. from_ym, to_ym으로 범위 지정
```
python merge_extract.py --type SOR_JIRA --input_path D:/data/SOR_JIRA --output_path D:/data/SOR_JIRA/sor_jira_merged.xlsx --from_ym 202006 --to_ym 202105
```
----
## C. Raw Data 읽어서 형태소 분리하여 말뭉치 생성 (추후 이를 사용하여 embedding생성)
### (non BERT용) 1-B에서 생성한 3개 파일(일부만 있어도 됨)를 입력으로 텍스트 전처리&형태소 분리하여 output_path에 지정한 파일에 쓴다. 가독성 때문에 개행 처리 
```
/* 실행 파라미터
  min_token_count : 문장의 최소 토큰 길이. 이 숫자 미만이면 문장의 조건으로 부적격하다고 판단하고 버림
  for_bert : bert용 파일을 만들지, non BERT용 파일을 만들지?
  is_all_tag : 모든 품사를 말뭉치에서 사용할지, 명사, 형용사 계열 등 토픽에 큰 영향을 미치는 품사만 사용할지?
*/

python preprocess_texts.py 
--input_path_sor ../TCL2021_Telco_Embedding_Dataset/merged/sor_merged.xlsx 
--input_path_sop ../TCL2021_Telco_Embedding_Dataset/merged/sop_merged.xlsx 
--input_path_sor_jira ../TCL2021_Telco_Embedding_Dataset/merged/sor_jira_merged.xlsx 
--output_path ../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora.dat 
--min_token_count 5 
--for_bert false 
--is_all_tag false
```  
### BERT 계열 모델 훈련시키기 위해서는 아래와 같은 구조로 입력 데이터가 만들어져 있어야 한다. 그리고 SOP(Sentence Order Prediction), NSP(Next Sentence Prediction) 등을 위해서는 최소 2개의 문장이 필요하다. 따라서 문서 1개에 문장이 1개만 있는 경우는 버린다.
```
문서1
문서1의 문장1
문서1의 문장2

문서2
문서2의 문장1
문서2의 문장2

문서3
...
```
### (BERT용) 1-B에서 생성한 3개 파일(일부만 있어도 됨)를 입력으로 텍스트 전처리&형태소 분리하여 output_path에 지정한 파일에 쓴다. 가독성 때문에 개행 처리

```
python preprocess_texts.py
--input_path_sor ../TCL2021_Telco_Embedding_Dataset/merged/sor_merged.xlsx
--input_path_sop ../TCL2021_Telco_Embedding_Dataset/merged/sop_merged.xlsx
--input_path_sor_jira ../TCL2021_Telco_Embedding_Dataset/merged/sor_jira_merged.xlsx
--output_path ../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora_for_bert.dat
--min_token_count 5
--for_bert true
--is_all_tag false
```
----  
# 2. SOP Dataset 생성
### 1-B에서 생성한 3개 파일 중 1개를 입력으로 downstream task에서 사용할 dataset을 생성한다. 현재는 SOP만 label 정제가 되어 있으며, SOR, SOR_JIRA도 생성은 가능하나 label 정제가 미흡하다.
```
python dataset_creator.py --type SOP --input_path D:/data/SOP --output_path ../TCL2021_Telco_Embedding_Dataset/dataset/sop_dataset.xlsx --from_ym 202002 --to_ym 202101
```  
----  
# 3. Embedding
## A. Word2Vec (gensim)
### 1-C에서 만든 말뭉치 중 단어 빈도수 top 10,000여개를 대상으로 Word2Vec Embedding을 생성한다.

- Embedding 차원수 : 128, 256, 384

- Window 크기 : 3, 4, 5

### 총 9개의 조합으로 각각 50 epochs를 돌려서 총 9개의 Word2Vec Embedding을 생성한다.
```
python word2vec_creator.py
```

## B. FastText (gensim)
### 1-C에서 만든 말뭉치 중 단어 빈도수 top 10,000여개를 대상으로 FastText Embedding을 생성한다. 단어를 10,000여개 골랐지만 FastText 특성 상 훨씬 더 많은 단어가 만들어진다.

- Embedding 차원수 : 128, 256, 384

- Window 크기 : 3, 4, 5

총 9개의 조합으로 각각 50 epochs를 돌려서 총 9개의 FastText Embedding을 생성한다.
```
python fasttext_creator.py
```
## C. AlBERT (pytorch)
### 1) 사전 파일 생성
```
/* BERT-pytorch 폴더 안에서 */
  python setup.py build
  python setup.py install

/* 실행 파라미터
  -c : 말뭉치 파일 경로
  -o : 결과물로 나올 사전 파일 경로
  -s : 단어 개수
  -m : 단어 최소 빈도수
*/
bert-vocab -c ../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora_for_bert.dat -o ../TCL2021_Telco_Embedding_Dataset/corpora/telco_vocab.dat -s 10000 -m 39
```
### 2) Pretraining - AlBERT 모델을 만들고 SOP, MLM을 수행하며 훈련한다.
```
/* 실행 파라미터
  -c : 말뭉치 파일 경로
  -v : 사전 파일 경로
  -o : 모델 파일 경로&접두어. epoch마다 ep? 가 뒤에 붙어서 저장된다.
  -l : encoder layer 수
  -a : multi head attention에서 head 수
  -s : 문장 내 토큰 최대 개수
  -hs : encoder layer의 hidden수
  -es : embedding layer의 hidden수
  -b : batch size
  -e : epochs
  -w : data loader worker 수
  -warmup_rate : 총 step 중 어느 정도 비율까지 warming up을 할 것인지? ex) 총 100,000step인데 0.05로 넣으면 5,000step까지 linear하게 lr 증가. 5001step부터 100,000step까지 linear하게 lr 감소
  -lr : learning rate
*/
bert -c ../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora_for_bert.dat -v ../TCL2021_Telco_Embedding_Dataset/corpora/telco_vocab.dat -o ../TCL2021_Telco_Embedding_Dataset/albert_model/albert.model -l 8 -a 8 -s 64 -hs 256 -es 128 -b 64 -e 10 -w 4 --warmup_rate 0.05 --lr 1e-3
```
----  
# 4. Downstream task (SOP 처리 부서 분류기)
3에서 만든 Embedding을 fine tuning하여 
## A. Word2Vec / FastText
```
sop_classifier.ipynb 열어서 순서대로 실행하면 총 9개의 embedding weights를 읽어 sop분류를 수행한다.
Embedding layer에 해당 embedding weights를 넣어준다.
embedding layer뒤쪽에는 Bidirectional LSTM -> FC -> Dropout -> FC -> Dropout -> FC -> softmax 로 구성되어 있다.
```
## B. AlBERT
### 아직 원인 파악이 안되었는데, jupyter notebook에서 pytorch기반의 모델 훈련이 불가한 상황. 어쩔 수 없이 .py 파일로 작성
```
python -W ignore sop_classifier_bert.py 
    -v ../TCL2021_Telco_Embedding_Dataset/corpora/telco_vocab.dat 
    -m ../TCL2021_Telco_Embedding_Dataset/albert_model/albert.model_weightsonly.ep9 
    -c sop_clf_bert_model_checkpoint.pt 
    -d ../TCL2021_Telco_Embedding_Dataset/dataset/sop_dataset.xlsx 
    -t False 
    -e 4

/* 파라미터 설명
    -v : 사전 파일 경로
    -m : pretrain된 albert모델 경로 (SOP, MLM직전 layer까지만 필요)
    -c : 저장할 sop 분류기 모델 파일 
    -d : 훈련 / 테스트에 쓰일 SOP dataset파일
    -t : True면 train / validation / test로 나눠서 훈련
         False면 나누지 않고 전체 훈련
         -> 처음 훈련 시에는 옵션을 True로 줘서 최적 하이퍼파라미터와 epoch를 찾고, 그 후에는 False로 전체 훈련 
    -e : epoch

ALBERT layer 뒤쪽은 FC -> Dropout -> FC -> softmax로 구성되어 있다.
*/    
```
----  
# 5. Deploy & Online learning
## SOP분류기를 시험해볼 수 있는 간단한 web화면을 fastapi로 개발. 틀린 예측에 대해 바로 훈련할 수 있는 online learning기능 도입 (AlBERT로 pretrain한 모델을 finetuning한 SOP분류기 내장)
```
pip install fastapi
pip install uvicorn
```
```
uvicorn sop_clf_web_api:app --reload
```
- GET /classlist
  ### class 목록 리턴
- GET /predict
  ### SOP 문장을 입력하면 그에 해당되는 처리 부서와 확률을 top 5까지 보여준다.
- POST /train
  ### SOP 문장과 모델에서 예측한 class, 실제 class를 request body에 넣어주면 실제 class로 예측할 수 있게 online learning 수행