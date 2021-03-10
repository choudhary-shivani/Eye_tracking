import time
import pandas as pd
import numpy as np
from sklearn.model_selection.tests.test_split import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# t = ['nFix', 'FFD', 'GPT',
#        'TRT', 'fixProp']


class Train:
    def __init__(self, df, split_rat=0.1, cross_valid=10, test_df=None,
                 n_component=10):
        self.df = df
        self.ratio = split_rat
        if split_rat is None:
            # print("No split selected")
            assert test_df is not None
            self.traindf = self.df
            self.testdf = test_df   # trial data set
            # print("Length of the test set ", len(self.traindf),len( test_df))
        else:
            self.testdf, self.traindf = self.__test_train_split()
        if cross_valid is not None:
            self.cross = True
        else:
            self.cross = False
        self.cvalid = cross_valid
        self.component = n_component
        self.model = self.train()
        self.X = None
        self.Y = None
        self.trainmodel = None
        self.pred = None


    def get_labels(self, test_train=True):
        labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
        drop_labels = ['sentence_id',  'word',
                       'Unnamed: 0', 'otag', 'utag',
                       'crossreftime',  'GPT-FFD',  'TRT-GPT', 'pps']
        if test_train:
            df = self.traindf
        else:
            df = self.testdf
        df = df[df.columns[~df.columns.isin(drop_labels)]]
        self.Y = df[labels]
        self.X = df[df.columns[~df.columns.isin(labels)]]
        # print(self.Y.columns, self.X.columns.to_list())

    def train(self):
        self.get_labels()
        # print(self.X.columns.to_list())
        self.trainmodel = PLSRegression(n_components=self.component)
        return self.trainmodel.fit(self.X, self.Y)

    def MAE(self, field=None, loc=0):
        return mean_absolute_error(self.Y[field], np.ravel(self.pred)[loc::5])

    def test(self):
        self.get_labels(test_train=False)
        # print("Length of the test set ", len(self.X))
        self.pred = self.model.predict(self.X)
        return self.MAE('nFix', 0), self.MAE('FFD', 1), self.MAE('GPT', 2), \
               self.MAE('TRT', 3), self.MAE('fixProp', 4)

    def __test_train_split(self):
        tr, te = train_test_split(self.df, train_size=self.ratio)
        return tr, te

    def model_process(self):
        if not self.cross:
            self.cvalid = 1
        acc = [0] * 5
        for i in range(self.cvalid):
            # self.testdf, self.traindf = self.__test_train_split()
            self.train()
            nfix, FFD, GPT, TRT, fixProp = self.test()
            acc[0] += nfix
            acc[1] += FFD
            acc[2] += GPT
            acc[3] += TRT
            acc[4] += fixProp
        # print("Value after {} fold validation".format(self.cvalid),
        #       np.array(acc) / self.cvalid)
        return np.array(acc) / self.cvalid


if __name__ == '__main__':
    data = pd.read_csv('training_pos_tagged.csv')
    trial_df = pd.read_csv('trial_pos_tagged.csv')
    # print("Len", len(trial_df))
    all_val = {}

    for i in range(1, 60, 5):
        start = time.time()
        # print("Value for the n_comp ", i)
        tr = Train(data, cross_valid=None, test_df=trial_df, split_rat=None,
                   n_component=i)
        all_val[i] = tr.model_process()
        print("Time taken for {} components: ".format(i), time.time() - start)

    for i in all_val.keys():
        print("Value at {} is ".format(i), all_val[i])
