import os
import tqdm
import re
import pronouncing
import numpy as np
# import seaborn as sns
import pandas as pd
from util.tagger import postagger, stopwords, eos_pattern, punct, wordnet_
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from BERT import BERTembed


def pre_processing(file_prefix='training'):
    loaded_data = pd.read_csv(file_prefix + '_data.csv')
    no_items = len(loaded_data)
    print("length of loaded data ", len(loaded_data))
    # To make encoding uniform we will put test data as well. Then calculate
    # one hot encoding
    if file_prefix == 'trial' or file_prefix == 'test':
        extended = pd.read_csv('training_data.csv')
        loaded_data = pd.concat([loaded_data, extended], axis=0)
        loaded_data.set_index(pd.Index(range(len(loaded_data))), inplace=True)

    # print(loaded_data.columns, extended.columns)

    # a = lambda x: ' '.join(x['word'].to_list())
    # sentence_data = loaded_data.groupby('sentence_id').apply(a)
    # for i in sentence_data.to_list():
    #     print(i)

    val = loaded_data['word'].apply(postagger)
    # print (np.ravel(val))
    otag = [i[0] for i in np.ravel(val)]
    utag = [i[1] for i in np.ravel(val)]
    # print(otag, utag)
    loaded_data['otag'] = otag
    loaded_data['utag'] = utag
    #change the categorical utag to the one hot encoded value
    encoded = LabelEncoder().fit_transform(loaded_data['utag'])
    encoded = encoded.reshape(-1, 1)
    encoded_vector = OneHotEncoder(sparse=False).fit_transform(encoded)
    loaded_data = pd.concat(
        [loaded_data, pd.DataFrame(encoded_vector, columns=['utag_e_' +
         str(i) for i in range(12)])], axis=1)
    # Need to remove the hard coding from
    # the code
    print("Length of the loaded_data ", len(loaded_data))
    loaded_data['toklen'] = loaded_data['word'].apply(len)
    # loaded_data['crossreftime'] = loaded_data['TRT'] - loaded_data['FFD']
    # loaded_data['GPT-FFD'] = loaded_data['GPT'] - loaded_data['FFD']
    # loaded_data['TRT-GPT'] = loaded_data['TRT'] - loaded_data['GPT']
    if file_prefix == 'trial' or file_prefix == 'test':
        loaded_data = loaded_data[:no_items]
    print("Length of the loaded_data ", len(loaded_data) )
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
    an.to_csv(file_prefix + '_pos_tagged.csv')
    loaded_data = pd.read_csv(file_prefix + '_pos_tagged.csv')
    val = loaded_data['word'].apply(wordnet_)
    loaded_data['pps'] = val
    x = lambda tr: len(pronouncing.phones_for_word(tr)[0].split(' ')) if len(
        pronouncing.phones_for_word(tr)) > 0 else 0
    val = loaded_data['word'].apply(x)
    loaded_data['phonem'] = val
    loaded_data.to_csv(file_prefix + '_pos_tagged.csv', index=False)


if __name__ == '__main__':
    pre_processing(file_prefix='test')
