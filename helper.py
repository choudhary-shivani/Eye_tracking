import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error


def get_labels(df, drop_pps = True):
    labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
    drop_labels = ['sentence_id', 'word_id', 'word',
                   'Unnamed: 0', 'otag', 'utag',
                   'crossreftime', 'GPT-FFD', 'TRT-GPT', 'pps']

    if drop_pps:
        drop_labels.append('pps')
    df = df[df.columns[~df.columns.isin(drop_labels)]]
    Y = df[labels]
    X = df[df.columns[~df.columns.isin(labels)]]
    return X, Y


def helper():
    print("===Evaluation result for local model===")
    b = pd.read_csv('trial_pos_tagged.csv')
    X, Y = get_labels(b, drop_pps=False)
    for name in ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']:
        model = load_model('fm_0_' + name)
        mae = mean_absolute_error(np.ravel(model.predict(X)),Y[name].to_numpy())
        print("MAE for {} is {}".format(name, mae))


def helper_combined():
    print("===Evaluation result for combined model===")
    b = pd.read_csv('trial_pos_tagged.csv')
    X, Y = get_labels(b)
    a = pd.read_csv('training_pos_tagged.csv')
    m = a[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']].mean()
    s = a[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']].std()
    model = load_model('combined_model')
    val = model.predict(X)
    val = (val * s.to_numpy()) + m.to_numpy()
    val = np.ravel(val)
    for idx, name in enumerate(['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']):
        mae = mean_absolute_error(val[idx::5], Y[name])
        print("MAE for {} is {}".format(name, mae))
    # print(val[::5])


if __name__ == '__main__':
    # helper_combined()
    helper()
