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