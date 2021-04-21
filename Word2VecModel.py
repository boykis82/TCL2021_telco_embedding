import numpy as np
import nltk
from itertools import chain
from sklearn.preprocessing import normalize
from scipy.stats import truncnorm

from gensim.models import Word2Vec
import LossLogger as ll

UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'


class Word2VecModel(object):
    def __init__(self):
        pass
    
    def load(self, model_file_name):
        # load
        self.gensim_w2v_model = Word2Vec.load(model_file_name)
        # 후처리
        self._post_model_created()
        
    def create(self, in_corpus_file_name, out_model_file_name,
               max_vocab_size=10000, embedding_size=100,
               epochs=10, window=5, workers=3):
        # input파일에서 말뭉치 추출
        corpus = [sentence.strip().split(' ') 
                  for sentence in open(in_corpus_file_name, 'r', encoding='utf-8').readlines()]
        # 빈도수 상위 n위의 최소빈도수 구함 (word2vec 훈련 시 그 이하는 버리기 위함)
        min_freq_cnt = self._get_min_freq_count(corpus, max_vocab_size)
        print(f'{max_vocab_size}개의 단어 내에서 최소 빈도수는 {min_freq_cnt}입니다.')

        loss_logger = ll.LossLogger()
        # gensim word2vec call
        self.gensim_w2v_model = Word2Vec(corpus, 
                         size=embedding_size, 
                         workers=workers, 
                         min_count=min_freq_cnt,
                         sg=1, 
                         iter=epochs,
                         callbacks=[loss_logger],
                         compute_loss=True,
                         window=window)
        # 저장
        self.gensim_w2v_model.callbacks = None
        self.gensim_w2v_model.save(out_model_file_name)   
        # 후처리
        self._post_model_created()        
        
    def _post_model_created(self):
        self.vocab_size = self.gensim_w2v_model.wv.vectors.shape[0]
        self.embedding_size = self.gensim_w2v_model.wv.vectors.shape[1]
        self.index2word = self.gensim_w2v_model.wv.index2word
        # unk, pad 추가
        self.index2word.append( UNK_TOKEN )
        self.index2word.append( PAD_TOKEN )    
        
        self.weight = self.gensim_w2v_model.wv.vectors
        self._append_unk_pad_vectors()
        
        # cosine유사도 체크를 위해 normalize
        self.norm_weight = normalize(self.weight, norm='l2', axis=1)

        # word를 index로 변환
        self.word2index = {w:i for i, w in enumerate(self.index2word)}
        # 사전. word를 vector로 변환 - 안쓰네
        #self.dictionary = {w:v for w, v in zip(self.index2word, unit_w2v)}    
        
    def _get_truncated_normal(self, mean=0, sd=1, low=-1, upp=1):
        return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd )
        
    # unknown, padding 토큰 추가
    def _append_unk_pad_vectors(self):
        self.weight = np.append(self.weight, 
                         self._get_truncated_normal().rvs(self.embedding_size * 2).reshape(2, self.embedding_size), axis=0)            
        
    # 빈도수 상위 vocab_size 내에 존재하는 단어 중 최소 빈도수를 구함
    def _get_min_freq_count(self, sentences, max_freq_rank):
        fdist = nltk.FreqDist(chain.from_iterable(sentences))
        return fdist.most_common(max_freq_rank)[-1][1] # the count of the the top-kth word      
    
    # 입력으로 받은 형태소 배열에서 그에 대응되는 단어 index반환
    def get_words_indexes(self, corpora):
        return [self.word2index[t] if t in self.word2index else self.word2index[UNK_TOKEN] for t in corpora]        