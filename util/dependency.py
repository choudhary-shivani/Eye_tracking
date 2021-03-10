import re
import string
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
eos_pattern = re.compile('EOS', re.IGNORECASE)
punct = string.punctuation


def dependecy_parse(line):
    line = re.sub(eos_pattern, '', line)
    # line.translate(str.maketrans('', '', punct))
    tagged_val = nlp(line)
    print(tagged_val.to_json())
    displacy.serve(tagged_val, style='dep')
    # for token in tagged_val:
    #     pos = spacy.explain(token.tag_).split(',')[0]
    #     if pos != 'unknown' and pos != 'punctuation mark':
    #         print(token, pos)
    print(list(tagged_val[1].subtree))


if __name__ == '__main__':
    dependecy_parse('I think this world may not be the best place.')