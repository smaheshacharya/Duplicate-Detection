"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
"""

import _pickle as cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import re
stop_words = stopwords.words('english')


def clean_text(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"[a-z]+\-[a-z]+", "", text)
    text = re.sub(r"[a-z]+\-", "", text)
    text = re.sub(r"\-[a-z]+", "", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # text = text.split()

    return text

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    # model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model.wmdistance(s1, s2)


norm_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


data = pd.read_csv('questions.csv')
data = data.loc[50000:100000,:]
data.drop('qid1',axis=1, inplace=True)
data.drop("qid2",axis=1, inplace=True)


wh = ['where','why','what','who','whom','how','when']
for x in wh:
  if x in stop_words:
    stop_words.remove(x)

for s in data.head()['question1']:
  print(s,'\n')

data['question1'] = data.question1.apply(lambda x: clean_text(x))
data['question2'] = data.question2.apply(lambda x: clean_text(x))

for s in data.head()['question1']:
  print(s,'\n')

# Added Features.
data['word_overlap'] = [set(x[0].split()) & set(x[1].split()) for x in data[['question1','question2']].values]
data['common_word_cnt'] = data['word_overlap'].str.len()

data['text1_nostop'] = data['question1'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
data['text2_nostop'] = data['question2'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

data['word_overlap'] = [set(x[0].split()) & set(x[1].split()) for x in data[['text1_nostop','text2_nostop']].values]
data['common_nonstop_word_cnt'] = data['word_overlap'].str.len()


data['char_cnt_1'] = data['question1'].str.len()
data['char_cnt_2'] = data['question2'].str.len()
data['char_cnt_diff'] = (data['char_cnt_1'] - data['char_cnt_2'])**2
data['word_cnt_1'] = data['question1'].apply(lambda x: len(str(x).split()))
data['word_cnt_2'] = data['question2'].apply(lambda x: len(str(x).split()))
data['word_cnt_diff'] = (data['word_cnt_1'] - data['word_cnt_2'])**2

data['avg_word_size_1'] = data['char_cnt_1']/data['word_cnt_1']
data['avg_word_size_2'] = data['char_cnt_2']/data['word_cnt_2']
data['avg_word_size_diff'] = (data['avg_word_size_1'] - data['avg_word_size_2'])**2

text1 = list(data['question1'])
text2 = list(data['question2'])
corpus1 = ' '.join(text1)
corpus2 = ' '.join(text2)
corpus = corpus1.lower() + corpus2.lower()
import nltk
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()
corpus = lem.lemmatize(corpus, "v")
#corpus = stem.stem(corpus)

tags =  pos_tag(corpus.split())
nouns = [i[0] for i in tags if i[1] in ("NN", "NNS", "NNP", "NNPS")]

def count_common_nouns(var1, var2, var3):
  count = 0
  for i in var1:
      if (i in var2) & (i in var3): 
        count += 1 
  return count  


data['text1_lower'] = data['question1'].apply(lambda x: x.lower())
data['text2_lower'] = data['question2'].apply(lambda x: x.lower()) 
data['common_noun_cnt'] = [count_common_nouns(nltk.word_tokenize(lem.lemmatize(x[0],"v")),nltk.word_tokenize(lem.lemmatize(x[1], "v")), nouns) for x in data[['question1','question2']].values]


# Initial Features


data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
data['diff_len'] = data.len_q1 - data.len_q2
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((data.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

# cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
# cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)

data.to_csv('./all49_2.csv', index=False)
