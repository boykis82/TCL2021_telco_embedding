import multiprocessing

# my lib
import textlib as tl
import Word2VecModel as wv
import FastTextModel as ft

def create_multi_w2v_model(picked_model_index, params):
    for i, (model_name, max_vocab_size, embedding_size, window_size, epochs) in \
            enumerate(zip(params['MODEL_NAME'],
                          params['MAX_VOCAB_SIZE'],
                          params['EMBEDDING_SIZE'],
                          params['WINDOW_SIZE'],
                          params['EPOCHS'])):
        if picked_model_index == -1:
            pass
        elif picked_model_index != i:
            continue
        
        print(f'---- {i} 시작!! ----')
        w2v_model = wv.Word2VecModel()
        w2v_model.create(corpora_file_name, 
                           w2v_model_file_name_prefix + model_name, 
                           max_vocab_size=max_vocab_size, 
                           embedding_size=embedding_size,
                           epochs=epochs,
                           window=window_size,
                           workers=WORKERS)    

# 테스트로 하나만 만들자.

if __name__ == '__main__':
    WORKERS = multiprocessing.cpu_count() - 1

    # 말뭉치 경로
    corpora_prttag_file_name = '../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora.dat'
    corpora_alltag_file_name = '../TCL2021_Telco_Embedding_Dataset/corpora/telco_corpora_all_tag.dat'

    # 일부 tag(명사, 형용사 계열)로만 만든 embedding vector를 저장할 경로
    w2v_model_prttag_file_name_prefix = '../TCL2021_Telco_Embedding_Dataset/embedding_w2v/telco_w2v_'

    # 모든 tag로 만든 embedding vector를 저장할 경로
    w2v_model_alltag_file_name_prefix = '../TCL2021_Telco_Embedding_Dataset/embedding_w2v_alltag/telco_w2v_'

    # 모든 tag로 만든 fasttext embedding vector를 저장할 경로
    ft_model_file_name_prefix = '../TCL2021_Telco_Embedding_Dataset/embedding_fasttext/telco_ft_'

    # w2v 모든 형태소 사용
    w2v_model_file_name_prefix = w2v_model_alltag_file_name_prefix
    corpora_file_name = corpora_alltag_file_name

    # 여러개의 w2v 모델을 만들기 위한 table
    MODEL_COUNT = 9

    W2V_TRAIN_PARAMS = {
        'MODEL_NAME': ['V10000_E128_W3','V10000_E128_W4','V10000_E128_W5',
                    'V10000_E256_W3','V10000_E256_W4','V10000_E256_W5',
                    'V10000_E384_W3','V10000_E384_W4','V10000_E384_W5'],
        'MAX_VOCAB_SIZE': [10000] * MODEL_COUNT,
        'EMBEDDING_SIZE': [128,128,128, 256,256,256, 384,384,384],
        'WINDOW_SIZE' : [3,4,5, 3,4,5, 3,4,5],
        #'EPOCHS': [20, 30, 40,  20, 30, 40,  20, 30, 40]
        'EPOCHS': [50] * MODEL_COUNT
    }

    create_multi_w2v_model(-1, W2V_TRAIN_PARAMS)
