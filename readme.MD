# 말뭉치 생성하기 위함이므로 딱히 업무 구분 둬서 조회할 필요 없음

# 1. raw data download

## 1-1. SOR
> ### ITRM - 요청관리 - 요청서 조회
> > 등록일 : 한달 단위로 입력 (ex: 2021년 2월이면 2021-02-01 ~ 2021-02-28)
> 
> 조회 후 SOR폴더에 202102.xlsx로 저장  
----  
## 1-2. SOP
> ### serviceflow - 장애조회 
> > 고객사 : SKT계열 
> 
> > 서비스구분 : Application
> 
> > 발생일자 : 한달 단위로 입력 (ex: 2021년 2월이면 2021-02-01 ~ 2021-02-28)
> 
> 조회 후 SOP폴더에 202102.xlsx로 저장  
----  
## 1-3. 변경계획서
> ### ITRM - 변경괸리 - 변경계획서 조회 
> > 날짜 : 운영배포예정일을 한달 단위로 입력 (ex: 2021년 2월이면 2021-02-01 ~ 2021-02-28)
>
> 조회 후 변경계획서 폴더에 202102.xlsx로 저장  
----  
# 2. 프로그램 실행법
## 2-1. raw data merge
> 1에서 다운로드 받은 파일이 d:/data 폴더 내에 변경계획서, SOR, SOP 폴더에 있다고 가정

```
/* 월별 변경계획서를 merge한 후 필요한 컬럼만 추출하여 output_path 로 지정한 파일에 쓴다. from_ym, to_ym으로 범위 지정! */

python merge_extract.py --type CHG --input_path D:/data/변경계획서 --output_path D:/data/변경계획서/chg_merged.xlsx --from_ym 201805 --to_ym 202102
```
```
/* 월별 SOR을 merge한 후 필요한 컬럼만 추출하여 output_path 로 지정한 파일에 쓴다. from_ym, to_ym으로 범위 지정!  */

python merge_extract.py --type SOR --input_path D:/data/SOR --output_path D:/data/SOR/sor_merged.xlsx --from_ym 202101 --to_ym 202102
```    
```
/* 월별 SOP를 merge한 후 필요한 컬럼만 추출하여 output_path 로 지정한 파일에 쓴다. from_ym, to_ym으로 범위 지정!  */

python merge_extract.py --type SOP --input_path D:/data/SOP --output_path D:/data/SOP/sop_merged.xlsx --from_ym 202101 --to_ym 202102
```  

## 2-2. raw data읽어서 형태소 분리하여 말뭉치 생성 (추후 이걸 활용하여 embedding생성)
```
/* 2-1에서 생성한 3개 파일(일부만 있어도 됨)를 입력으로 텍스트 전처리&형태소 분리하여 output_path에 지정한 파일에 쓴다. */

python preprocess_texts.py --input_path_chg D:/data/변경계획서/chg_merged.xlsx --input_path_sor D:/data/SOR/sor_merged.xlsx --input_path_sop D:/data/SOP/sop_merged.xlsx --output_path D:/data/telco_corpora.dat
```  
----  
## 2-3. raw data읽어서 dataset생성 
```
/* 2-1에서 생성한 3개 파일(일부만 있어도 됨)를 입력으로 dataset 생성한다.  */

python dataset_creator.py --type SOR --input_path D:/data/SOR --output_path D:/data/SOR/sor_dataset.xlsx --from_ym 201706 --to_ym 202102
```  
---
# 3. 1/2항은 본인만의 데이터셋을 만들 때 참고하시면 될 것 같고요.
## 3-1. dataset 자료 설명
> 상기 1항/2항의 자료는 텔레콤망에서만 받을 수 있기 때문에 제가 dataset폴더에 총 3개의 파일을 압축하여 올려뒀습니다.
> > chg_merged.zip  --> 변경계획서 파일들을 201706 ~ 202102 까지 merge하여 필요한 컬럼만 추출한 자료입니다. (embedding생성에 사용)
---
> > sor_merged.zip  --> 요청서 파일들을 201706 ~ 202102 까지 merge하여 필요한 컬럼만 추출한 자료입니다. (embedding생성에 사용)
---
> > sor_dataset.zip --> 요청서 파일로부터 만든 dataset입니다. training data에 요청서 내용, 요청부서, 요청회사 등이 있으며 label은 SOR처리부서입니다.
---
## 3-2. 개요
> make_w2v.ipynb에 작성해뒀습니다. jupyter notebook으로 열어보시면 되고요. python 3.8 && tensorflow 2.4 입니다.
---
> 추가로 깔아야 하는 라이브러리가 많습니다. 로컬에서 돌리시려면 anaconda 환경에서 구축하시는 걸 권해드립니다.
---
> 에러 나면 뭐 깔라고 에러메시지 나오니까 참고하시면 됩니다.
---
> 주석을 대강 달아두긴 했는데 혹시 설명이 필요하시면 나중에 시간 잡아서 설명드리겠습니다.
---
>> 크게 2step으로 나눠져 있습니다.
---
## 3-3. word2vec생성(upstream task)
> gensim 라이브러리를 사용하였습니다. Word2VecModel이라는 클래스를 만들었습니다. 2항의 output을 입력으로 embedding을 만듭니다.
---
> 여러 하이퍼파라미터를 줄 수 있습니다. 저는 베이스라인 모델로 [총 단어수 10,000개+a, 임베딩 100차원, window size 3, epochs 40)으로 돌렸습니다.
---

## 3-4. deep learning 모델 훈련(downstream task)
> 2-3으로 만든 dataset을 로딩&전처리하고, tensorflow로 3-3에서 만든 임베딩 가중치를 내포한 SORClassffier에 입력으로 집어넣어 분류작업을 수행합니다.
---
> 레이어 조금만 쌓아서 가볍게 돌렸는데 80%정도 정확도를 보이고 있습니다. 전처리를 좀 더 잘하고 모델 더 다듬으면 더 좋은 결과도 가능할 것 같습니다.