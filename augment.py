import pandas as pd
import numpy as np
import tqdm
from BERT import BERTembed
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from util.tagger import postagger, stopwords, eos_pattern, punct, wordnet_

def encode(col):
    encoded = LabelEncoder().fit_transform(col)
    encoded = encoded.reshape(-1, 1)
    encoded_vector = OneHotEncoder(sparse=False).fit_transform(encoded)
    df = pd.DataFrame(encoded_vector)
    return df


bol = lambda bo: 1 if bo else 0


def field(f, p):
    # print("dtyp", pd.api.types.infer_dtype(p[f]))
    if p[f].dtype == bool:
        return p[f].apply(bol)
        # pass
    elif p[f].dtype == object:
        return encode(p[f])
    else:
        return p[f]


def file_aug(type='train'):
    p = pd.read_csv(type + '_ling_features.csv')
    # a = pd.read_csv(type + '_pos_tagged.csv')
    x = p.columns.to_list()
    # print(p.dtypes)
    for i in ['sentence_id', 'word_id', 'word']:
        x.remove(i)
    for ifl in x:
        # print(, ifl)
        val = field(ifl, p)
        shp = val.shape[-1]
        # print(shp)
        val.columns = [ifl + '_' + str(i) for i in range(shp)]
        if p[ifl].dtype == object:
            del p[ifl]
            p = pd.concat([p, val], axis=1)
        else:
            p[ifl] = val
            print(p[ifl])

    loaded_data = p
    print("Length of the loaded_data ", loaded_data.shape )
    all_embedding = np.array([])
    temp = np.array([])
    for idx, word in enumerate(tqdm.tqdm(loaded_data['word'].to_list())):
        # print("Processing word no ", str(idx))
        if idx % 1000 == 0:
            print("Appending the big array")
            all_embedding = np.append(all_embedding, temp)
            temp = np.array([])
        word = word.translate(str.maketrans('', '', punct))
        word = re.sub(eos_pattern, '', word)
        if word != '':
            val = BERTembed(word)
            # all_embedding.append(val)
        else:
            # print("Processing word no/ Missing index ", str(idx))
            val = np.zeros(768)
        temp = np.append(temp, val)
    all_embedding = np.append(all_embedding, temp)
    an = loaded_data[:]
    d = pd.DataFrame(np.reshape(all_embedding, (-1, 768)))
    an = pd.concat([an, d], axis=1)
    # used for the visualization purpose
    an.to_csv(type + '_pos_tagged_ling.csv')
    # print(a, a.shape)


def data_loader():
    an = pd.read_csv('combo_pos_tagged_ling.csv')
    trial = pd.read_csv('trial_data.csv')
    train = pd.read_csv('training_data.csv')


if __name__ == '__main__':
    file_aug('combo')
