import pandas as pd
from tqdm import tqdm_notebook as tqdm
import spacy

nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner", "textcat", "entity_ruler", "merge_noun_chunks",
                                            "merge_entities", "merge_subtokens"])
df = pd.read_pickle('../input/jigsaw-unintended-bias-in-toxicity-classification/folds.pkl')
df2 = pd.read_csv('../input/old_toxic/train.csv')

corpus = df2['comment_text'].values.tolist()
i = 0
with open('../input/lm_corpus.txt', 'w+') as f:
    for doc in tqdm(nlp.pipe(corpus, n_threads=16, batch_size=128)):
        for sentence in doc.sents:
            ss = str(sentence).strip()
            if len(ss) > 0:
                f.write(ss + '\n')
        f.write('\n')
df3 = pd.read_csv('../input/old_toxic/test.csv')
corpus = df3['comment_text'].values.tolist()
i = 0
with open('../input/lm_corpus.txt', 'a+') as f:
    for doc in tqdm(nlp.pipe(corpus, n_threads=16, batch_size=128)):
        for sentence in doc.sents:
            ss = str(sentence).strip()
            if len(ss) > 0:
                f.write(ss + '\n')
        f.write('\n')
