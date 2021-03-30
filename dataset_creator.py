import pandas as pd
import os
import argparse

from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta

EXTRACT_COLS = {
    'SOR' : ['제목', '요청부서', '고객사', '요청사유', '요청내역', '서비스유형(중)']
    #'SOP' : ['장애제목', '조치 내역', '작업처리내용']
    #'SOP' : ['서비스']
}


# 입력 argeunemt를 parsing하여 dictionary 형태로 반환
'''
    python dataset_creator.py --type SOR --input_path D:/data/SOR --output_path D:/data/SOR/sor_dataset.xlsx --from_ym 201706 --to_ym 202102
    python dataset_creator.py --type SOP --input_path D:/data/SOP --output_path D:/data/SOP/sop_dataset.xlsx --from_ym 201706 --to_ym 202102
'''
def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='월 별 파일에서 분류 모델 구축을 위한 dataset 생성')
    arg_parser.add_argument('--type', help='SOP(SOP), SOR(요청서) ...', type=str)
    arg_parser.add_argument('--input_path', help='월 별 파일이 위치한 경로. 각 파일은 yyyymm.xlsx 포맷이어야 합니다.', type=str)
    arg_parser.add_argument('--output_path', help='dataset 파일이 생성될 경로(파일명 포함)', type=str)
    arg_parser.add_argument("--from_ym", help="from date : YYYYMM", type=str)
    arg_parser.add_argument("--to_ym", help="to date : YYYYMM", type=str)        
    args = arg_parser.parse_args()    

    return args
    

# from_date 부터 to_date까지 yyyymm 포맷의 날짜 리스트 반환
def enumerate_dates(from_ym, to_ym):
    i = 0
    mths = []
    while True:
        ym = dt.strptime(from_ym, '%Y%m') + relativedelta(months=i)
        mths.append(ym.strftime('%Y%m'))   
        if ym == dt.strptime(to_ym, '%Y%m'):
            break
        i += 1         
    return mths    


# from_date 부터 to_date까지의 파일(yyyymm.xls)을 pandas dataframe 포맷으로 merge
def create_dataset(args):    
    dfs = []

    print(f'{args.type} dataset 생성 시작!')

    # 월별로 파일 읽어서 필요한 컬럼만 추출하여 dfs 에 append
    for ym in enumerate_dates(args.from_ym, args.to_ym):
        print(f'{ym} 처리중...')
        input_file_name = os.path.join(args.input_path, f'{ym}.xlsx')
        try:
            df = pd.read_excel(input_file_name, sheet_name=0, engine='openpyxl')
        except FileNotFoundError:
            print(f'{input_file_name}이 없습니다! skip!')
            continue

        # 필요한 컬럼만 추출
        
        df = df[EXTRACT_COLS[args.type]]
        df['req_ym'] = ym
        df['co'] = df['고객사']
        df['req_br'] = df['요청부서']
        df['sentence'] = df['제목'] + ' . ' + df['요청사유'] + ' . ' + df['요청내역']
        df['label'] = df['서비스유형(중)'] 

        df.drop(['제목', '요청부서', '고객사', '요청사유', '요청내역', '서비스유형(중)'], axis=1, inplace=True)

        dfs.append(df)

    # merge
    all_df = pd.concat(dfs)   

    # output file에 merge된 자료 쓴다.
    print(f'{args.type} dataset 파일 생성 시작... {args.output_path}')

    with pd.ExcelWriter(args.output_path) as writer:
        all_df.to_excel(writer, 'Sheet1', index=False, engine='openpyxl')
        writer.save()

    print(f'{args.type} dataset 파일 생성 완료! {args.output_path}')


if __name__ == '__main__':
    # 입력 argument parsing
    args = parse_arguments()
    create_dataset(args)
