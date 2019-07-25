import pickle
from pathlib import Path

import numpy as np

identity_words = [
        set(['gay','homosexual','lesbian','lgbt','bisexual','heterosexual','bisexual','homosexuals']),
        set(['muslim','islam','islamic','muslims']),
        set(['jewish','christian','palestinian','jew','jews','church','christians','christianity','catholics','catholic']),
        set(['psychiatric','mental'])
    ]


def get_num_features(text):
    retval = np.zeros(12)
    retval[0] = len(text)
    retval[1] = len(set(text))
    retval[2] = len(set(text))/retval[0]
    for w in text:
        retval[11]+=len(w)
        temp = w.lower()
        for idx, word_set in enumerate(identity_words):
            if temp in word_set:
                retval[idx + 3]+=1
                retval[idx + 7] += 1/retval[0]
    return retval


def main():

    data_dir = Path('../input/preparedRNNData')
    with open(data_dir / 'training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)

    VECTORIZER_PATH = Path('../input/preparedRNNData/cntv.pkl')
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    feature1 = np.vstack(list(map(get_num_features, training_data)))
    feature2 = vectorizer.transform(list(map(lambda x: ' '.join(x), training_data))).toarray()
    features = np.hstack([feature1, feature2])

    with open(data_dir/'features.npy','wb') as f:
        np.save(f, features)


if __name__ == '__main__':
    main()