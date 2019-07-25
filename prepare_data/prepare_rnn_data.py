import os
import time
import numpy as np
import pandas as pd
import gensim
from tqdm import tqdm
from nltk.stem import PorterStemmer

ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()
from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")
import gc
import pickle
import re
import spacy
from pathlib import Path

spell_model = gensim.models.KeyedVectors.load_word2vec_format('../input/wikinews300d1mvec/wiki-news-300d-1M.vec')
words = spell_model.index2word
w_rank = {}
for i, word in enumerate(words):
    w_rank[word] = i
WORDS = w_rank


# Use fast text as vocabulary
def words(text): return re.findall(r'\w+', text.lower())


def load_myemb(word_dict, lemma_dict):
    EMBEDDING_FILE = '../input/myembedding/myembedding.pkl'
    with open(EMBEDDING_FILE, 'rb') as f:
        embeddings_index = pickle.load(f)
    embed_size = 100
    nb_words = len(word_dict) + 10
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    print(unknown_vector[:5])
    for key in tqdm(word_dict):
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lemma_dict[key]
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        # if len(key) > 1:
        #     word = correction(key)
        #     embedding_vector = embeddings_index.get(word)
        #     if embedding_vector is not None:
        #         embedding_matrix[word_dict[key]] = embedding_vector
        #         continue
        embedding_matrix[word_dict[key]] = unknown_vector
    return embedding_matrix, nb_words


def load_glove(word_dict, lemma_dict):
    EMBEDDING_FILE = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
    with open(EMBEDDING_FILE, 'rb') as f:
        embeddings_index = pickle.load(f)
    embed_size = 300
    nb_words = len(word_dict) + 10
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    print(unknown_vector[:5])
    for key in tqdm(word_dict):
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lemma_dict[key]
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        if len(key) > 1:
            word = correction(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
        embedding_matrix[word_dict[key]] = unknown_vector
    return embedding_matrix, nb_words


def load_crawl(word_dict, lemma_dict):
    EMBEDDING_FILE = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
    with open(EMBEDDING_FILE, 'rb') as f:
        embeddings_index = pickle.load(f)
    embed_size = 300
    nb_words = len(word_dict) + 10
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    print(unknown_vector[:5])
    for key in tqdm(word_dict):
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lemma_dict[key]
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        if len(key) > 1:
            word = correction(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
        embedding_matrix[word_dict[key]] = unknown_vector
    return embedding_matrix, nb_words

start_time = time.time()
print("Loading data ...")
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
train_text = train['comment_text']
test_text = test['comment_text']
text_list = pd.concat([train_text, test_text])

from sklearn.feature_extraction.text import CountVectorizer
cntv = CountVectorizer(ngram_range=(1, 1), min_df=1e-4, token_pattern=r'\w+',analyzer='char')
cntv.fit(text_list)
import pickle
with open('../input/preparedRNNData/cntv.pkl','wb') as f:
    pickle.dump(file=f, obj=cntv)

y = train['target'].values
num_train_data = y.shape[0]
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("Spacy NLP ...")
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
word_dict = {}
word_index = 2  # 0 for pad token, 1 for unknown
lemma_dict = {}
docs = nlp.pipe(text_list, n_threads=8, batch_size=128)
word_sequences = []
for doc in tqdm(docs):
    word_seq = []
    for token in doc:
        if token.is_space: continue
        if token.like_url:
            t = 'url'
        elif token.like_num:
            t = '100'
        elif token.like_email:
            t = 'email'
        else:
            t = token.text

        if (t not in word_dict) and (token.pos_ is not "PUNCT"):
            word_dict[t] = word_index
            word_index += 1
            lemma_dict[t] = token.lemma_
        if token.pos_ is not "PUNCT":
            word_seq.append(t)
    while len(word_seq) < 2: word_seq.append('.')
    word_sequences.append(word_seq)
del docs
gc.collect()
train_word_sequences = word_sequences[:num_train_data]
test_word_sequences = word_sequences[num_train_data:]

save_dir = Path('../input/preparedRNNData')
save_dir.mkdir(exist_ok=True, parents=True)

with open(save_dir / 'word_dict.pkl', 'wb') as f:
    pickle.dump(word_dict, f)
with open(save_dir / 'lemma_dict.pkl', 'wb') as f:
    pickle.dump(lemma_dict, f)
with open(save_dir / 'training_data.pkl', 'wb') as f:
    pickle.dump(train_word_sequences, f)
print("--- %s seconds ---" % (time.time() - start_time))

from gensim.models import Word2Vec

NewdataEmbedding = Word2Vec(word_sequences, size=100, window=5, min_count=1, workers=8, iter=5, seed=2019)  # 训练100维的

output_dir = Path('../input/myembedding')
output_dir.mkdir(exist_ok=True, parents=True)

embedding_dict = {w: NewdataEmbedding.wv[w] for w in NewdataEmbedding.wv.vocab.keys()}

with open(output_dir / 'myembedding.pkl', 'wb+') as f:
    pickle.dump(embedding_dict, f)
print("--- %s seconds ---" % (time.time() - start_time))

glove_embedding_matrix_glove, nb_words = load_glove(word_dict, lemma_dict)
print(nb_words)
crawl_embedding_matrix_glove, nb_words = load_crawl(word_dict, lemma_dict)
my_embedding_matrix, nb_words = load_myemb(word_dict, lemma_dict)

print("--- %s seconds ---" % (time.time() - start_time))

with open(save_dir / 'glove_embedding.npy', 'wb') as f:
    np.save(f, glove_embedding_matrix_glove)

with open(save_dir / 'crawl_embedding.npy', 'wb') as f:
    np.save(f, crawl_embedding_matrix_glove)
with open(save_dir / 'my_embedding.npy', 'wb') as f:
    np.save(f, my_embedding_matrix)
