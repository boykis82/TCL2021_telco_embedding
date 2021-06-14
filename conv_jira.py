import pandas as pd
import os
import argparse
from bs4 import BeautifulSoup

from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta
from pandas.core.frame import DataFrame

COLS = ['summary', 'created', 'description', 'components']

'''
jira issue 검색조건
  project = OP301 AND issuetype = SOR요청서  and created >= '2020-06-01' and created <= '2020-06-30' and component is not null order by created

jira issue 추출컬럼
  요약, 생성됨, 설명, 구성 요소, 요청자 부서
'''

# python conv_jira.py --input_path D:/data/SOR_JIRA_BF --output_path D:/data/SOR_JIRA --from_ym 202006 --to_ym 202105
def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='HTML로 되어 있는 jira 자료를 excel로 변환')
    arg_parser.add_argument('--input_path', help='월 별 파일이 위치한 경로. 각 파일은 yyyymm.html 포맷이어야 합니다', type=str)
    arg_parser.add_argument('--output_path', help='변환된 파일이 생성될 경로(파일명 포함)', type=str)
    arg_parser.add_argument("--from_ym", help="from date : YYYYMM", type=str)
    arg_parser.add_argument("--to_ym", help="to date : YYYYMM", type=str)        

    return arg_parser.parse_args()    


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


# from_date 부터 to_date까지의 파일(yyyymm.html)을 pandas dataframe 포맷으로 merge
def convert_jira_from_html_to_excel(args):
    print(f'jira 변환 시작!')

    for ym in enumerate_dates(args.from_ym, args.to_ym):
        df = pd.DataFrame()

        print(f'{ym} 처리중...')
        input_file_name = os.path.join(args.input_path, f'{ym}.html')
        
        try:
            with open(input_file_name, encoding='UTF8') as fp:
                soup = BeautifulSoup(fp, 'html.parser')

                for col in COLS:
                    data = soup.find_all('td', {'class':col}) 
                    print(f'   {col} 추출중... {len(data)}건 찾음.') 
                    data = [d.get_text().strip() for d in data]
                    df[col] = data

        except FileNotFoundError:
            print(f'{input_file_name}이 없습니다!')
            continue             

        except ValueError:
            raise

        output_file_name = os.path.join(args.output_path, f'{ym}.xlsx')
        with pd.ExcelWriter(output_file_name) as writer:
            df.to_excel(writer, 'Sheet1', index=False, engine='openpyxl')
            writer.save()        

if __name__ == '__main__':
    # 입력 argument parsing
    args = parse_arguments()
    convert_jira_from_html_to_excel(args)
