import pandas as pd
import os
import textlib as tl
import argparse

try:
    from eunjeon import Mecab
except:
    from konlpy.tag import Mecab

class InvalidInputException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


# 입력 argeunemt를 parsing하여 dictionary 형태로 반환
'''
    python preprocess_texts.py --input_path_chg ../TCL2021_Telco_Embedding_Dataset/merged/chg_merged.xlsx --input_path_sor ../TCL2021_Telco_Embedding_Dataset/merged/sor_merged.xlsx --input_path_sop ../TCL2021_Telco_Embedding_Dataset/merged/sop_merged.xlsx --input_path_sor_jira ../TCL2021_Telco_Embedding_Dataset/merged/sor_jira_merged.xlsx --output_path ../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora_all_tag.dat --min_token_count 5 --split_between_documents false --is_all_tag true
    python preprocess_texts.py --input_path_chg ../TCL2021_Telco_Embedding_Dataset/merged/chg_merged.xlsx --input_path_sor ../TCL2021_Telco_Embedding_Dataset/merged/sor_merged.xlsx --input_path_sop ../TCL2021_Telco_Embedding_Dataset/merged/sop_merged.xlsx --input_path_sor_jira ../TCL2021_Telco_Embedding_Dataset/merged/sor_jira_merged.xlsx --output_path ../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora_for_bert_all_tag.dat --min_token_count 5 --split_between_documents true --is_all_tag true
'''
def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='텍스트 clenasing & 형태소 나누기')
    arg_parser.add_argument('--input_path_chg', help='merge된 변경계획서 파일이 위치한 경로', type=str)
    arg_parser.add_argument('--input_path_sor', help='merge된 sor 파일이 위치한 경로', type=str)
    arg_parser.add_argument('--input_path_sop', help='merge된 sop 파일이 위치한 경로', type=str)
    arg_parser.add_argument('--input_path_sor_jira', help='merge된 jira sor 파일이 위치한 경로', type=str)
    arg_parser.add_argument('--output_path', help='전처리 후 merge된 파일이 생성될 경로(파일명 포함)', type=str)    
    arg_parser.add_argument('--min_token_count', help='문장의 토큰 개수가 몇 개 이하면 무시할지?', type=int)    
    arg_parser.add_argument('--split_between_documents', help='문서 간 CR 줄 것인지?(bert는 CR필요)', type=str)
    arg_parser.add_argument('--is_all_tag', help='모든 형태소를 취급할것인가?', type=str)
    args = arg_parser.parse_args()    

    return args


# 여러 컬럼으로 되어 있는 텍스트를 하나의 컬럼으로 만들기 위해 vertical 방향으로 쌓는다.
def stack_texts_vertically(df):
    '''
           col1    col2
    row1    1       2  
    row2    3       4

    형태로 된 자료를

          
    1
    3
    2
    4

    형태로 반환한다.
    '''
    concat_df = pd.concat( [df.iloc[:,i] for i in range(df.shape[1])] )
    return concat_df.drop_duplicates(keep='first', inplace=False)


# from_date 부터 to_date까지의 파일(yyyymm.xls)을 pandas dataframe 포맷으로 merge
def preprocess_texts(args):    
    input_file_paths = [f for f in [args.input_path_chg, args.input_path_sop, args.input_path_sor, args.input_path_sor_jira] if f is not None]
    split_between_docs = True if args.split_between_documents.lower() == 'true' else False
    is_all_tag = True if args.is_all_tag.lower() == 'true' else False

    if len(input_file_paths) == 0:
        raise InvalidInputException('입력파일이 최소 1개는 있어야 합니다!')

    mecab = Mecab()

    with open(args.output_path, 'w', encoding='utf-8') as fo:    
        for input_file_path in input_file_paths:
            
            print('-' * 80)
            print(f'{input_file_path} 처리 시작...')

            try:
                df = pd.read_excel(input_file_path, sheet_name=0, engine='openpyxl')
            except FileNotFoundError:
                raise InvalidInputException('입력 파일이 없습니다! 경로를 확인해주세요.')

            texts = stack_texts_vertically(df)  
            print(f'    {texts.shape[0]} 개 존재.(중복 제거 후)')

            for i, text in enumerate(texts):
                try:
                    cleansed_text = tl.clean_text(text)
                except TypeError:
                    print(f'      {i+1} 번째 데이터에 문제가 있어 skip!')
                    continue

                sentences = tl.segment_sentences(cleansed_text)
        
                # 은전한닢 Mecab 형태소 분석기로 문장들을 잘라 파일에 쓴다. 토큰 개수가 min_token_count 이하면 훈련에서 제외.
                tl.write_corpora(sentences, fo, args.min_token_count, mecab, is_all_tag)
                # bert는 문장단위로 CR줘야 훈련 가능!
                if split_between_docs:
                    fo.write('\n')

                if i % 5000 == 0 and i > 0:
                    print(f'      {i} 번째 데이터 처리 완료!')

            print(f'{input_file_path} 처리 완료!')

if __name__ == '__main__':
    # 입력 argument parsing
    args = parse_arguments()
    preprocess_texts(args)
