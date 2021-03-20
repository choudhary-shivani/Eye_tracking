import torch
import pandas as pd
import spacy
import syllables
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

# fill file name here
data_df = pd.read_csv('training_data.csv')
word_new = []

for i in range(len(data_df['word'])):
    word_new.append(data_df['word'][i].rstrip('.<EOS>').strip())
data_df['word'] = word_new
data_df.dropna(inplace=True)
col = data_df.columns.tolist()
target_cols = [item for item in col if item not in ['sentence_id', 'word', 'word_id']]
y = data_df[target_cols].to_numpy()
np.save('target.npy', y)

sentences = []
count = data_df.sentence_id.value_counts()
counter = 0
for i in range(int(data_df['sentence_id'].tolist()[0]), int(data_df['sentence_id'].tolist()[-1]+1)):
    list_sent = []
    for j in range(counter, count.get(key=i) + counter):
        list_sent.append(data_df.word[j])
    sentence = ' '.join(list_sent)
    # Removing extra white spaces
    sentences.append(' '.join(sentence.strip().split()))
    counter = counter + count.get(key=i)


nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


# A function to get spacy features
def get_features_spacy(tok):
    if tok.ent_type_ != '':
        return [tok.text, tok.pos_, tok.tag_, syllables.estimate(tok.text), tok.is_stop, tok.ent_type_,
                tok.dep_]
    else:
        return [tok.text, tok.pos_, tok.tag_, syllables.estimate(tok.text), tok.is_stop, "UNKNOWN",
                tok.dep_]


# A function to get Bert Embeddings
def to_bert_embeddings(text, return_tokens=False):
    tokens = tokenizer.tokenize(text)
    tokens_with_tags = ['[CLS]'] + tokens + ['[SEP]']
    indices = tokenizer.convert_tokens_to_ids(tokens_with_tags)
    out = model(torch.LongTensor(indices).unsqueeze(0))
    embeddings_matrix = torch.stack(out[0]).squeeze(1)[-4:]
    embeddings = []
    for k in range(embeddings_matrix.shape[1]):
        mat_em = embeddings_matrix[:, k, :]
        embeddings.append(torch.sum(mat_em[-4:], dim=0).detach().numpy())
    embeddings = embeddings[1:-1]
    if return_tokens:
        assert len(embeddings) == len(tokens)
        return embeddings, tokens
    else:
        return embeddings, tokens


spacy_features = []
for sentence in sentences:
    sentence_features = []
    doc = nlp(sentence)
    for token in doc:
        token_f = np.hstack(get_features_spacy(token))
        sentence_features.append(token_f)
    spacy_features.append(np.vstack(sentence_features))

# Creating numpy array of features from spaCy and Syllables
features_array = np.concatenate(spacy_features)

# Creating and storing word embeddings
embeddings_bert_list = []
bert_tokens = []
for sentence in sentences:
    sent_embeddings, tokens_sent_bert = to_bert_embeddings(sentence, return_tokens=False)
    embeddings_bert_list.append(np.vstack(sent_embeddings))
    bert_tokens.append(tokens_sent_bert)

# Creating the array of Bert Embeddings
embeddings_bert_array = np.concatenate(embeddings_bert_list, axis=0)
bert_tokens_numpy_array = np.hstack(bert_tokens)

# Creating list of sentence IDs
counter = 0
sentence_id_list = []
for sentence in sentences:
    doc = nlp(sentence)
    for token in doc:
        sentence_id_list.append(counter)
    counter = counter + 1

sentence_ids_array = np.array(sentence_id_list)

# Creating column names of dataframe of all features
column_names = ['sentence_id', 'word', 'pos_tag', 'detailed_tag', 'syllables', 'is_stop', 'ner_tag', 'dependency']

data = np.column_stack((sentence_ids_array, features_array))
df_features = pd.DataFrame(data, columns=column_names)
df_features.to_csv("Features_Spacy_.csv")

embeddings_df = pd.DataFrame(np.column_stack((bert_tokens_numpy_array, embeddings_bert_array)))
embeddings_df.to_csv("BERT_Embedding_.csv")


df_bert_raw = pd.read_csv('BERT_Embedding_.csv')
df_bert_raw.rename(columns={'0': 'words'}, inplace=True)
df_bert_raw.drop(['Unnamed: 0'], axis=1, inplace=True)

indices_of_nan = []
for i, w in enumerate(df_bert_raw['words'].tolist()):
    if str(w) == 'nan':
        indices_of_nan.append(i)

df_nan = df_bert_raw.iloc[indices_of_nan, 1:]
df_nan.reset_index(inplace=True)
df_nan.drop(['index'], axis=1, inplace=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_tokens(phrase):
    tokens_ = tokenizer.tokenize(phrase)
    length_tok = len(tokens_)
    return tokens_, length_tok


def get_word_embeddings(tok, length_tok, len_df):
    bert = []
    df_fn = df_bert_raw.iloc[len_df: len_df + length_tok, :]
    df_fn = df_fn.reset_index()
    try:
        df_fn.drop(['Unnamed: 0'], axis=1, inplace=True)
    except KeyError:
        pass
    for k in range(length_tok):
        try:
            idx = df_fn['words'].to_list().index(str(tok[k]))
            bert_array = df_fn.iloc[idx, 2:]
            bert.append(bert_array.to_numpy())
            print(bert_array.shape)
            df_fn.drop(idx)
            df_fn.reset_index()
            try:
                df_fn.drop(['Unnamed: 0'], axis=1, inplace=True)
            except KeyError:
                pass

        except ValueError:
            print("Value Error")
            bert_array = df_nan.iloc[0]
            print(bert_array.shape)
            try:
                df_nan.drop(0, inplace=True).reset_index(inplace=True).drop(['index'], axis=1, inplace=True)
            except AttributeError:
                pass
            bert.append(bert_array.to_numpy())
            df_fn.reset_index()
            try:
                df_fn.drop(['Unnamed: 0'], axis=1, inplace=True)
            except KeyError:
                pass

    bert_array_phrase = np.row_stack(bert)
    return np.mean(bert_array_phrase, axis=0)


bert_word_wise_embeddings = []
word_list = []
length_covered = 0

for word in data_df['word']:
    tokens, len_tok = get_tokens(phrase=word)
    if word == ' ' or word == '':
        bert_word_wise_embeddings.append(np.zeros(768))
    else:
        bert_word_wise_embeddings.append(get_word_embeddings(tokens, len_tok, length_covered))
    length_covered = length_covered + len_tok

word_list = data_df['word']


embeddings_array = np.row_stack(bert_word_wise_embeddings)
embeddings_df = pd.DataFrame(np.column_stack((word_list, embeddings_array)))
embeddings_df.to_csv("BERT_phrase_embeddings.csv")


def get_features(phrase, len_phrase, len_cov):
    df_fn = df_features.iloc[len_cov: len_cov + len_phrase, :]
    df_fn = df_fn.reset_index()
    df_fn.drop('index', axis=1, inplace=True)
    syll_tot = 0
    count = 0
    for i, token in enumerate(phrase):
        # w = token.text
        if not token.is_punct:
            count = count + 1
            # idx = df_fn['word'].tolist().index(w)
            syll = int(df_fn.loc[i]['syllables'])
            syll_tot = syll_tot + syll
            pos_tag_ = df_fn.loc[i]['pos_tag']
            dep_tag_ = df_fn.loc[i]['dependency']
            det_tag_ = df_fn.loc[i]['detailed_tag']
            ner_tag_ = df_fn.loc[i]['ner_tag']
            stop_tag_ = df_fn.loc[i]['is_stop']
            df_fn.drop(i, inplace=True)
            df_fn.reset_index()
    if count == 0:
        pos_tag_ = df_fn.loc[len_phrase-1]['pos_tag']
        dep_tag_ = df_fn.loc[len_phrase-1]['dependency']
        det_tag_ = df_fn.loc[len_phrase-1]['detailed_tag']
        ner_tag_ = df_fn.loc[len_phrase-1]['ner_tag']
        stop_tag_ = df_fn.loc[len_phrase-1]['is_stop']
        return [0, pos_tag_, dep_tag_, det_tag_, ner_tag_, stop_tag_]
    else:
        return [syll_tot, pos_tag_, dep_tag_, det_tag_, ner_tag_, stop_tag_]


length_covered = 0
feature_list = []
hyphen_feature = []
non_alnum_feature = []
for token in data_df['word']:
    doc = nlp(token)
    num_non_alnum = 0
    w = [token.text for token in doc]
    if '-' in w:
        hyphen_ = 1
    else:
        hyphen_ = 0
    for word in doc:
        if word.is_punct:
            num_non_alnum = num_non_alnum + 1

    phrase_length = len(doc)
    if phrase_length > 1:
        list_f = get_features(doc, phrase_length, length_covered)

        feature_list.append(list_f)
    elif phrase_length == 1 and num_non_alnum == 1:
        df_new = df_features.loc[length_covered]

        feature_list.append([0, df_new['pos_tag'], df_new['dependency'], df_new['detailed_tag'],
                             df_new['ner_tag'], df_new['is_stop']])
    else:
        df_new = df_features.loc[length_covered]

        feature_list.append([df_new['syllables'], df_new['pos_tag'], df_new['dependency'], df_new['detailed_tag'],
                             df_new['ner_tag'], df_new['is_stop']])
    hyphen_feature.append(hyphen_)
    non_alnum_feature.append(num_non_alnum)
    length_covered = length_covered + phrase_length
hyphen_feature = np.array(hyphen_feature)
non_alnum_feature = np.array(non_alnum_feature)
sub_feature_array = np.row_stack(feature_list)
sentence_id_array = np.array(data_df['sentence_id'])
word_array = np.array(data_df['word'])
word_id_array = np.array(data_df['word_id'])
column_names = ['sentence_id', 'word', 'word_id', 'syllables', 'pos', 'dep', 'detailed', 'ner', 'is_stop', 'hyphen', 'non_alnum']


data = np.column_stack((sentence_id_array, word_array, word_id_array, sub_feature_array, hyphen_feature, non_alnum_feature))
df_linguistic = pd.DataFrame(data, columns=column_names)
df_linguistic.columns = column_names
df_linguistic.to_csv('Linguistic_Features.csv')


bert_phrase_df = pd.read_csv("BERT_phrase_embeddings.csv")
bert_phrase_df.drop(['Unnamed: 0'], axis=1, inplace=True)
bert_phrase_df.drop(['0'], axis=1, inplace=True)

# bert names
bert_col = []
for i in range(768):
    col = "bert" + "_" + str(i + 1)
    bert_col.append(col)
col = data_df.columns.tolist()
target_cols = [item for item in col if item not in ['sentence_id', 'word', 'word_id']]

# For test data, remove target_cols in the statement below
column_names = [*column_names, *bert_col, *target_cols]

# For test data remove data_df[target_cols] in the statement below
final_df = pd.concat([df_linguistic, bert_phrase_df, data_df[target_cols]], axis=1)
final_df.columns = column_names
final_df.astype({'syllables': 'float64'})
# saving the combined Dataframe to CSV
final_df.dropna(inplace=True)
final_df.reset_index(inplace=True)
final_df.drop(['index'], axis=1, inplace=True)
final_df.to_csv("Full_Feature_Space.csv")

# We can drop remaining columns and keep only syllables and bert columns to get features with only syllables
syllables_col = ['syllables', *bert_col, *target_cols]
final_df.to_csv("Syllables_Feature_Space.csv")

# Syllables features
syllables_bert_features = final_df[['syllables', *bert_col]].to_numpy()
np.save('syllables_bert_features.npy', syllables_bert_features)
syllables_bert_target = final_df[target_cols].to_numpy()
np.save('syllables_bert_target.npy', syllables_bert_target)

