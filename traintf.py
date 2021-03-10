import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection.tests.test_split import train_test_split
from sklearn.metrics import mean_absolute_error
import keras.backend as K
from custom_error import mae_
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.device('cpu:0')


class Traintf:
    def __init__(self, df, split_rat=0.1, cross_valid=10, test_df=None,
                 batch_size=32, epochs=80):
        self.df = df
        self.ratio = split_rat
        if split_rat is None:
            # print("No split selected")
            assert test_df is not None
            self.traindf = self.df
            self.testdf = test_df
            print("Length of the test set ", len(self.traindf), len(test_df))
        else:
            self.testdf, self.traindf = self.__test_train_split()

        if cross_valid is not None:
            self.cross = True
        else:
            self.cross = False
        self.cvalid = cross_valid
        self.X = None
        self.Y = None
        self.trainmodel = None
        self.pred = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.i = None
        self.model = None

    def get_labels(self, test_train=True):
        labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
        drop_labels = ['sentence_id', 'word',
                       'Unnamed: 0', 'otag', 'utag',
                       'crossreftime', 'GPT-FFD', 'TRT-GPT', 'pps',
                       'phonem']

        if test_train:
            df = self.traindf
        else:
            df = self.testdf
        df = df[df.columns[~df.columns.isin(drop_labels)]]
        self.Y = df[labels]
        self.X = df[df.columns[~df.columns.isin(labels)]]

    def lr_rate(self, epoch, lr):
        if epoch > 30:
            lr = lr * tf.math.exp(-0.01)
        else:
            lr = 1e-5
        return lr

    def norma(self, x):
        return (x - x.min()) / (x.max() - x.min()), x.min().to_numpy(), \
               x.max().to_numpy()

    def train(self, fields=None, load_previous=False,
              old=None, crnt=0):
        assert fields is not None
        if isinstance(fields, str):
            fields = [fields]
        # val = {}
        self.get_labels(test_train=False)
        X_test, Y_test = self.X, self.Y
        self.get_labels()
        print("Test size", X_test.shape)
        print("training size ", self.X.shape)
        for field in fields:
            print("Traing the NN for field - ", field)
            callback = tf.keras.callbacks.LearningRateScheduler(self.lr_rate,
                                                                verbose=False)
            if not load_previous:
                print("==============Building models=============")
                input = tf.keras.Input(shape=(self.X.to_numpy().shape[-1]),
                                       name='embed')
                x = layers.Dense(1024, activation='relu')(input)
                x = layers.Dropout(rate=0.2, seed=10)(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.Dropout(rate=0.2, seed=10)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dense(256, activation='relu')(x)
                nfix = layers.Dropout(rate=0.2)(x)
                nfix = layers.Dense(64, activation='relu', name='nfix0')(nfix)
                # nfix = (layers.Dense(64, activation='relu', name='nfix0',
                #                      kernel_regularizer='l2')(x))
                nfix = layers.Dense(16, activation='relu', name='nfix2')(nfix)
                nfix = layers.Dense(1, activation='relu', name='NFIX')(nfix)
                self.model = Model(input, [nfix])
                self.model.compile(optimizer='adam',
                                   loss='mae'
                                   )
            else:
                print("==============Loading models=============")
                self.model = load_model("temp_model_" + field)
            print(self.model.summary(),
                  self.X.columns.to_list()[:20])
            keras.backend.set_value(self.model.optimizer.learning_rate, 1e-5)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=False,
                               patience=30)
            mc = ModelCheckpoint('fm_' + str(crnt) + "_" + field,
                                 monitor='val_loss',
                                 mode='min', verbose=0, save_best_only=True)
            self.model.fit(self.X.to_numpy(),
                           self.Y[field].to_numpy(),
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           verbose=True,
                           # validation_split=0.2,
                           validation_data=(X_test.to_numpy(),
                                            Y_test[field].to_numpy()),
                           callbacks=[callback, mc, es],
                           use_multiprocessing=True)
            self.model.save("temp_model_" + field)

    def test(self, fields=None):
        assert fields is not None
        self.get_labels(test_train=False)
        val = {}
        print(self.X.shape, self.X.columns.to_list())
        x = pd.DataFrame()
        for idx, field in enumerate(t):
            print("Test data size ", len(self.X))
            model = load_model("fm_0_" + field)
            v = model.predict(self.X.to_numpy())
            # print(v)
            x[field] = np.ravel(v)
            print("Mae for {}".format(field),
                  mean_absolute_error(v, np.ravel(self.Y)[idx::5]))
            # original_val = tr.Y['nFix'].to_list()
            # pred_val = np.ravel(tr.pred)[::5]
        # print("Metrics is ", val)
        return x

    def __test_train_split(self):
        tr, te = train_test_split(self.df, train_size=self.ratio)
        return tr, te

    def model_process(self, fields):
        if not self.cross:
            self.cvalid = 1
        final_val = pd.DataFrame()
        for i in range(self.cvalid):
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Processing cross validation iteration {}".format(i))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            # if i == 0:
            #     self.train(fields, load_previous=False, old=None, crnt=i)
            # else:
            #     self.train(fields, load_previous=True, old=i-1, crnt=i)
            fv = self.test(fields)
            final_val = pd.concat([final_val, fv], axis=1)
        # print(final_val, "Total val: ", pd.DataFrame(final_val).mean(axis=1))
        # print("Inference val: ", final_val)
        return final_val


def get_labels(df):
    labels = ['crossreftime', 'GPT-FFD', 'TRT-GPT']
    drop_labels = ['sentence_id', 'word_id', 'word',
                   'Unnamed: 0', 'otag', 'utag',
                   'pps',
                   'nFix', 'FFD', 'GPT', 'TRT', 'fixProp']

    # if test_train:
    #     df = self.traindf
    # else:
    #     df = self.testdf
    df = df[df.columns[~df.columns.isin(drop_labels)]]
    Y = df[labels]
    X = df[df.columns[~df.columns.isin(labels)]]
    return X, Y


if __name__ == '__main__':
    data = pd.read_csv('training_pos_tagged.csv')
    trial_df = pd.read_csv('test_pos_tagged.csv')
    trial_df = trial_df.fillna(0)
    # X, _ = get_labels(trial_df)
    # # for col in ['crossreftime',  'GPT-FFD',  'TRT-GPT']:
    # col = 'combo'
    # model = load_model('fm_0_' + col)
    # val = model.predict(X)
    # intermediate_from_a = Model(model.input,
    #                             model.get_layer('NFIX').output)
    # val = intermediate_from_a.predict(X)
    # print(intermediate_from_a.predict(X))
    #
    # trial_df[['crossreftime',  'GPT-FFD',  'TRT-GPT']] = val
    # print(trial_df[['crossreftime',  'GPT-FFD',  'TRT-GPT']])
    # print("Trial df size", trial_df.shape)
    # X, _ = get_labels(data)
    # # for col in ['crossreftime',  'GPT-FFD',  'TRT-GPT']:
    # #     model = load_model('fm_0_' + col)
    # #     val = model.predict(X)
    # data[['crossreftime',  'GPT-FFD',  'TRT-GPT']] = intermediate_from_a.predict(X)
    # print(data[['crossreftime',  'GPT-FFD',  'TRT-GPT']])
    # print("Trial df size", data.shape)
    t = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
    # t = ['nFix']
    tr = Traintf(data, cross_valid=1, test_df=trial_df, split_rat=None,
                 batch_size=32, epochs=80)
    # # tr.train(fields=t)
    # # tr.get_labels()
    infrence_val = tr.model_process(fields=t)
    final_mat = pd.concat([trial_df[['sentence_id', 'word_id',
                                     'word']], infrence_val], axis=1)
    final_mat.to_csv('inference_NN_final.csv', index=False)
    # # tr.train()
