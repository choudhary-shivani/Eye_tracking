import re
import os
import nltk
import string
import numpy as np
import pandas as pd
from math import log
from nltk.corpus import stopwords
from nltk.corpus import wordnet
# from util.tagger import postagger


Stopwords = set(stopwords.words('english'))
punct = string.punctuation
eos_pattern = re.compile('EOS', re.IGNORECASE)


def postagger(word):
    if isinstance(word, str):
        word = word.translate(str.maketrans('', '', punct))
        word = re.sub(eos_pattern, '', word)
        if word != '':
            word = [word]
            tag = nltk.pos_tag(word)[0][1]
            return tag, nltk.mapping.map_tag('en-ptb', 'universal', tag)
        else:
            return 'OTHE', 'OTHE'


def wordnet_(name):
    out = {}

    for synset in wordnet.synsets(name):
        name_ = synset.lemmas()[0].name()
        if name not in out.keys():
            out[name] = {
                "v": [],
                "n": [],
                "a": [],
                "r": []
            }

        if synset.pos() == 'v':
            out[name]['v'].append(name_)
        elif synset.pos() == 'n':
            out[name]['n'].append(name_)
        elif synset.pos() == 'a':
            out[name]['a'].append(name_)
        elif synset.pos() == 'r':
            out[name]['r'].append(name_)

        # for lemma in synset.lemmas():
        #     syn.append(lemma)  # add the synonyms
        #     if lemma.antonyms():  # When antonyms are available, add them into the list
        #         ant.append(lemma.antonyms()[0].name())
        # print('Synonyms: ' + str(syn))
        # print('Antonyms: ' + str(ant))
    # print(out)
    lval = 1
    if len(out) > 0:
        for key, val in out[name].items():
            length = len(np.unique(val))
            if length > 0:
                lval *= 1 / length
            # print(key, np.unique(val))
    return -lval*log(lval)

if __name__ == '__main__':
    postagger(['I', 'think', 'I', 'like', 'you'])