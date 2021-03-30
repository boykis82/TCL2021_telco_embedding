import pandas as pd
import os
import argparse

from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta

EXTRACT_COLS = {
    'CHG' : ['제목', '변경내용'],
    'SOR' : ['제목', '요청사유', '요청내역'],
    'SOP' : ['장애제목', '조치 내역', '작업처리내용']
}


# 입력 argeunemt를 parsing하여 dictionary 형태로 반환
'''
    python merge_extract.py --type CHG --input_path D:/data/변경계획서 --output_path D:/data/변경계획서/chg_merged.xlsx --from_ym 201805 --to_ym 202102
    python merge_extract.py --type SOR --input_path D:/data/SOR --output_path D:/data/SOR/sor_merged.xlsx --from_ym 202101 --to_ym 202102
    python merge_extract.py --type SOP --input_path D:/data/SOP --output_path D:/data/SOP/sop_merged.xlsx --from_ym 202101 --to_ym 202102
'''
def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='월 별 파일에서 embedding 생성에 필요한 컬럼만 추출한 뒤 전체 월 자료 merge하여 결과 파일 생성')
    arg_parser.add_argument('--type', help='CHG(변경계획서), SOP(SOP), SOR(요청서) ...', type=str)
    arg_parser.add_argument('--input_path', help='월 별 파일이 위치한 경로. 각 파일은 yyyymm.xlsx 포맷이어야 합니다.', type=str)
    arg_parser.add_argument('--output_path', help='merge 완료된 파일이 생성될 경로(파일명 포함)', type=str)
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
def merge(args):    
    dfs = []

    print(f'{args.type} merge 시작!')

    # 월별로 파일 읽어서 필요한 컬럼만 추출하여 dfs 에 append
    for ym in enumerate_dates(args.from_ym, args.to_ym):
        print(f'{ym} 처리중...')
        input_file_name = os.path.join(args.input_path, f'{ym}.xlsx')
        try:
            df = pd.read_excel(input_file_name, sheet_name=0, engine='openpyxl')
        except FileNotFoundError:
            continue
        # 변경계획서는 헤더가 2줄이라 1개 날려야 함
        if args.type == 'CHG':
            df.drop([0], inplace=True)
        # 필요한 컬럼만 추출
        df = df[ EXTRACT_COLS[args.type] ]
        dfs.append(df)

    # merge
    all_df = pd.concat(dfs)   

    # output file에 merge된 자료 쓴다.

    with pd.ExcelWriter(args.output_path) as writer:
        all_df.to_excel(writer, 'Sheet1', index=False, engine='openpyxl')
        writer.save()

    print(f'{args.type} merge 완료! {args.output_path}')


if __name__ == '__main__':
    # 입력 argument parsing
    args = parse_arguments()
    merge(args)
