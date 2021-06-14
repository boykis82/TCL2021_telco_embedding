import konlpy
from konlpy.tag import Okt
from konlpy.tag import Kkma
from konlpy.tag import Komoran
try:
    import eunjeon
    from eunjeon import Mecab
except:
    from konlpy.tag import Mecab
    
import regex
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# 문서를 문장으로 분리
def segment_sentences(text):
    return [w.strip() for w in regex.split("([.?!])?[\n]+|[.?!] ", text) if w is not None and len(w.strip()) > 1]

# 문장 클렌징
def clean_text(text):
    text = strip_tags(text) # remove html tags
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    
    text = regex.sub(u"[^ \r\n\p{Latin}\p{Hangul}_.?/!]", " ", text) # Replace unacceptable characters with a space.

    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    
    return text

# 문장을 tokenize 하여 corpus를 파일에 쓴다.
def write_corpora(sentences, output_file_handle, tagger=None):
    if tagger is None:
        tagger = Mecab()    
    target_tags = get_tags(tagger)
    
    for i, s in enumerate(sentences):
        try:
            pos_tagged = tagger.pos(s)               
        except ValueError:
            print(f'could not {i}th parsed! sentence = {s}')
            continue

        tokenized = [t[0].strip() for t in pos_tagged if t[1] in target_tags]
        output_file_handle.write(' '.join(tokenized) + '\n')    
        

# 문장을 tokenize 하여 return
def get_corpora(sentences, ignore_words, tagger=None):
    if tagger is None:
        tagger = Mecab()
    target_tags = get_tags(tagger)
    
    corporas = []
    for i, s in enumerate(sentences):
        try:
            pos_tagged = tagger.pos(s)               
        except ValueError:
            print(f'could not {i}th parsed! sentence = {s}')
            continue

        tokenized = [t[0].strip() for t in pos_tagged if t[1] in target_tags and t[0] not in ignore_words]
        corporas.append( ' '.join(tokenized) )
        
    return corporas       

# 내가 원하는 형태소만 취사선택
def get_tags(tagger):
    if isinstance(tagger, konlpy.tag._okt.Okt):
        return ['Alpha', 'Noun', 'Adjective']
    elif isinstance(tagger, konlpy.tag._kkma.Kkma):
        return ['NN', 'NNG', 'NNB', 'NNM',' NNP', 'NP', 'NR', 'OH', 'OL', 'ON', 'VA', 'VXA']
    elif isinstance(tagger, konlpy.tag._komoran.Komoran):
        return ['NNG', 'NNB', 'NNP', 'NP', 'NR', 'SH', 'SL', 'SN', 'VA']
    else:
        if isinstance(tagger, eunjeon._mecab.Mecab) or isinstance(tagger, konlpy.tag._mecab.Mecab):
            return ['VA', 'NNG', 'NNB', 'NNBC', 'NNP', 'NP', 'NR', 'SH', 'SL', 'SN', 'VA']
        else:
            raise ValueError(f'invalid tagger!! {tagger.__class__}')

