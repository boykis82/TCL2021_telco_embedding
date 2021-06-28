import pandas as pd
import os
import argparse

from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta

EXTRACT_COLS = {
    'SOR' : ['제목', '요청부서', '고객사', '요청사유', '요청내역', '서비스유형(중)'],
    'SOP' : ['고객사', '장애제목', '상세내역', '서비스모듈', '담당BA부서', '장애발생일시'],
    'SOR_JIRA' : ['summary', 'description', 'components']
}


# 입력 argeunemt를 parsing하여 dictionary 형태로 반환
'''
    python dataset_creator.py --type SOR_JIRA --input_path D:/data/SOR_JIRA --output_path D:/data/SOR_JIRA/sor_jira_dataset.xlsx --from_ym 202006 --to_ym 202105
    python dataset_creator.py --type SOR --input_path D:/data/SOR --output_path D:/data/SOR/sor_dataset.xlsx --from_ym 201706 --to_ym 202105
    python dataset_creator.py --type SOP --input_path D:/data/SOP --output_path D:/data/SOP/sop_dataset.xlsx --from_ym 202002 --to_ym 202101
'''
def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='월 별 파일에서 분류 모델 구축을 위한 dataset 생성')
    arg_parser.add_argument('--type', help='SOP(SOP), SOR(요청서), SOR_JIRA(JIRA요청서) ...', type=str)
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

        # Swing관련된 SOP만 추출      
        if args.type == 'SOP':  
            df = df[(   (df['고객사'] == 'SK텔레콤') | 
                        (df['고객사'] == 'SK브로드밴드')
                    )
                    &
                    (   df['서비스 구분'] == 'Application'  )
                    &
                    (   (df['담당BA부서'] == '고객채널Unit') |
                        (df['담당BA부서'] == 'Billing Unit') |
                        (df['담당BA부서'] == '고객상품Unit') |
                        (df['담당BA부서'] == '통신MIS Unit') |
                        (df['담당BA부서'] == '고객서비스팀') |
                        (df['담당BA부서'] == 'ICT혁신팀') |
                        (df['담당BA부서'] == 'Digital Billing팀') |
                        (df['담당BA부서'] == 'Broadband사업팀') |
                        (df['담당BA부서'] == '채널혁신팀') |
                        (df['담당BA부서'] == 'ICT혁신팀') |
                        (df['담당BA부서'] == 'T Biz. Digital그룹') |
                        (df['담당BA부서'] == 'ICT Biz. Digital그룹'))]

        # 필요한 컬럼만 추출
        df = df[EXTRACT_COLS[args.type]]
        if args.type == 'SOR':
            df['req_ym'] = ym
            df['co'] = df['고객사']
            df['req_br'] = df['요청부서']
            df['sentence'] = df['제목'] + ' . ' + df['요청사유'] + ' . ' + df['요청내역']
            df['label'] = df['서비스유형(중)'] 

            df.drop(['제목', '요청부서', '고객사', '요청사유', '요청내역', '서비스유형(중)'], axis=1, inplace=True)
            
        elif args.type == 'SOP':
            df['date'] = df['장애발생일시'].apply(slice_date_only)
            
            df['co'] = df['고객사']
            df['sentence'] = df['장애제목'] + ' . ' + df['상세내역']
            df['label_org'] = df['서비스모듈']
            df['label_clean'] = df['서비스모듈'].apply(conv_label)
            df['label_clean'] = df['label_clean'].apply(conv_label_etc)

            df.drop(['고객사', '장애제목', '상세내역', '서비스모듈', '장애발생일시'], axis=1, inplace=True)

        elif args.type == 'SOR_JIRA':
            df['req_ym'] = ym
            df['co'] = 'SKT'
            df['sentence'] = df['summary'] + ' . ' + df['description']
            df['label'] = df['components'] 

            df.drop(['summary', 'description', 'components'], axis=1, inplace=True)            

        dfs.append(df)

    # merge
    all_df = pd.concat(dfs)   

    # output file에 merge된 자료 쓴다.
    print(f'{args.type} dataset 파일 생성 시작... {args.output_path}')

    with pd.ExcelWriter(args.output_path) as writer:
        all_df.to_excel(writer, 'Sheet1', index=False, engine='openpyxl')
        writer.save()

    print(f'{args.type} dataset 파일 생성 완료! {args.output_path}')

def slice_date_only(datetime_):
    return datetime_[0:8]

def conv_label(label_):
    if label_ == 'SWING  Payment' or label_ == 'SWING  수납' or label_ == 'SWING 자납' or label_ == 'SWING 수납' or label_ == 'SWING 재무/정산':
        return 'SWING Payment'
    elif label_ == 'SWING오더 - 번호이동/명변/해지/복지/제휴카드/보증보험' or label_ == 'SWING오더 - 고객/청구정보/정지/통계' or \
        label_ == 'SWING 오더 - 가입/기변/할부/보조금' or label_ == 'SWING오더 - 무선오더/사업개발' :
        return 'SWING 오더 - 무선오더'
    elif label_.find('멤버십') >= 0:
        return 'SWING 멤버십'
    elif label_ == 'SKB  SWING Portal':
        return 'SWING Portal'        
    elif label_.find('CTC') >= 0:        
        return 'SWING CTC'
    elif label_ == 'B스마트플래너' or label_ == '스마트 플래너' or label_ == '스마트플래너 MMS 모바일웹':
        return 'SWING 스마트플래너'
    else:
        return label_

def conv_label_etc(label_):
    if label_ != 'SWING 오더 - 무선오더' and \
        label_ != '상품-무선' and \
        label_ != 'SWING Payment' and \
        label_ != 'SWING 청구' and \
        label_ != 'SWING 유선오더' and \
        label_ != 'SWING CTC-SKT' and \
        label_ != 'SWING 시설' and \
        label_ != 'SWING 유선상품' and \
        label_ != '상품-단말기' and \
        label_ != 'SWING 파트너관리(PRM)' and \
        label_ != 'SWING 스마트플래너' and \
        label_ != '상품-Interface' and \
        label_ != 'SWING 주소' and \
        label_ != 'SWING 자원 - 계약서관리' and \
        label_ != '인터페이스(SMS,MMS포함)' and \
        label_ != 'SWING 개통' and \
        label_ != 'T gate' and \
        label_ != 'SWING 단말' and \
        label_ != 'SWING 미납' and \
        label_ != 'SWING 멤버십' and \
        label_ != 'SWING CTC-SKB' and \
        label_ != 'SWING 모바일' and \
        label_ != '과금정보' and \
        label_ != '판매점 SSO' and \
        label_ != 'SWING 장애' and \
        label_ != 'UI, 프레임웍' and \
        label_ != 'SWING DBM' and \
        label_ != 'SWING SSO' and \
        label_ != 'SWING NIS' and \
        label_ != 'MVNO' and \
        label_ != 'MPAMS' and \
        label_ != 'SWING 재무/정산' and \
        label_ != 'SWING Portal' and \
        label_ != '소액결제-ISAS' and \
        label_ != 'SWING 유선OSS 모바일' and \
        label_ != '서식지통합관리' and \
        label_ != '과금계산' and \
        label_ != 'SKT eService-Tsales':
        return "기타"
    else:
        return label_



if __name__ == '__main__':
    # 입력 argument parsing
    args = parse_arguments()
    create_dataset(args)
