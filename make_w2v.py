import pandas as pd
import os
import argparse

from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta




# 입력 argeunemt를 parsing하여 dictionary 형태로 반환
'''
    python dataset_creator.py --type SOR --input_path D:/data/SOR --output_path D:/data/SOR/sor_dataset.xlsx --from_ym 201706 --to_ym 202102
    python dataset_creator.py --type SOP --input_path D:/data/SOP --output_path D:/data/SOP/sop_dataset.xlsx --from_ym 201706 --to_ym 202102
'''
def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='말뭉치 파일로부터 파라미터에 맞는 w2v생성')
    arg_parser.add_argument('--input_path', help='말뭉치 파일 경로. 파일이 위치한 경로. 각 파일은 yyyymm.xlsx 포맷이어야 합니다.', type=str)
    arg_parser.add_argument('--output_path', help='word2vec 파일이 생성될 경로(파일명 포함)', type=str)
    arg_parser.add_argument('--max_vocab_size', help='사전 내에 존재할 최대 단어 수(너무 크면 overfitting발생 가능', type=int, default=10000)
    arg_parser.add_argument('--embedding_size', help='임베딩 차원수', type=int, default=200)
    arg_parser.add_argument('--multi_cpu_count', help='embedding생성 시 동원할 cpu개수', type=int, default=1)        
    arg_parser.add_argument('--window_size', help='embedding생성 시 단어 전후로 몇개까지 볼 것인지?', type=int, default=5)        
    arg_parser.add_argument('--epochs', help='training epochs', type=int, default=15)        
    args = arg_parser.parse_args()    

    return args
    
if __name__ == '__main__':
    # 입력 argument parsing
    args = parse_arguments()
    create_dataset(args)
