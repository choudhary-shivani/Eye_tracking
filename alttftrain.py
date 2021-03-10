import os
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection.tests.test_split import train_test_split
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_absolute_error
from custom_error import mae_, custom_mae
import keras.backend as K
from functools import partial, update_wrapper
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.device('cpu:0')
tf.compat.v1.enable_eager_execution()


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
        train_len = len(self.traindf)
        commond_df = pd.concat([self.traindf, self.testdf])
        self.m, self.s, commond_df = self.norma(commond_df)
        with open('store', 'w') as f:
            f.write(str(self.m) + ',' + str(self.s))
        self.traindf = commond_df[:train_len]
        self.testdf = commond_df[train_len:]
        # print(self.testdf['nFix'])
        self.testdf.to_csv('temp_pos.csv')
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
        self.X_test = None
        self.Y_test = None

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

    def lr_rate(self, epoch, lr):
        if epoch > 30:
            lr = lr * tf.math.exp(-0.01)
        else:
            lr = 1e-5
        return lr

    def norma(self, a):
        m = a[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']].mean()
        s = a[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']].std()
        p = lambda x: (x - m.to_numpy(dtype=np.float32)) / s.to_numpy(
            dtype=np.float32)
        a[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = a[
            ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']].T.apply(p).T
        return m, s, a

    def train(self, fields=None, load_previous=False):
        assert fields is not None
        if isinstance(fields, str):
            fields = [fields]
        # val = {}
        self.get_labels(test_train=False)
        X_test, Y_test = self.X, self.Y
        self.X_test, self.Y_test = self.X, self.Y
        self.get_labels()
        print("Test size", X_test.shape)
        print("training size ", self.X.shape)
        # for field in fields:
        #     print("Traing the NN for field - ", field)
        callback = tf.keras.callbacks.LearningRateScheduler(self.lr_rate,
                                                            verbose=False)
        err = partial(custom_mae, s=self.s, m=self.m)
        err = update_wrapper(err, custom_mae)
        if not load_previous:
            print("==============Building models=============")
            input = tf.keras.Input(shape=(self.X.to_numpy().shape[-1]),
                                   name='embed')
            # x = layers.Dense(2048, activation='relu')(input)
            # x = layers.Dropout(rate=0.3, seed=10)(x)
            x = layers.Dense(1024, activation='relu',
                             kernel_initializer='random_normal',
                             bias_initializer='zeros'
                             )(input)
            x = layers.Dense(512, activation='relu',
                             kernel_initializer='random_normal',
                             bias_initializer='zeros'
                             )(x)
            x = layers.Dropout(rate=0.2, seed=10)(x)
            x = layers.Dense(256, activation='relu',
                             kernel_initializer='random_normal',
                             bias_initializer='zeros'
                             )(x)
            nfix = layers.Dropout(rate=0.2)(x)
            nfix = layers.Dense(64, activation='relu', name='nfix0',
                                kernel_initializer='random_normal',
                                bias_initializer='zeros'
                                )(nfix)
            # nfix = (layers.Dense(64, activation='relu', name='nfix0',
            #                      kernel_regularizer='l2')(x))
            nfix = layers.Dense(16, activation='relu', name='nfix2',
                                kernel_initializer='random_normal',
                                bias_initializer='zeros'
                                )(nfix)
            nfix = layers.Dense(5,  name='NFIX',
                                kernel_initializer='random_normal',
                                bias_initializer='zeros'
                                )(nfix)
            self.model = Model(input, [nfix])
            self.model.compile(optimizer='adam',
                               loss='mae'
                               # run_eagerly=True
                               )
        # else:
        #     print("==============Loading models=============")
        #     self.model = load_model("temp_model_" + field)
        self.model.summary()
        keras.backend.set_value(self.model.optimizer.learning_rate, 1e-5)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=False,
                           patience=30)
        mc = ModelCheckpoint('fm_master', monitor='val_loss',
                             mode='min', verbose=0, save_best_only=False)
        # print(self.Y, Y_test, self.X, X_test)
        self.model.fit(self.X.to_numpy(),
                       self.Y.to_numpy(),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=True,
                       validation_data=(X_test.to_numpy(),
                                        Y_test.to_numpy()),
                       callbacks=[callback, mc, es, PredictionCallback()],
                       use_multiprocessing=True)
        val = self.model.predict(X_test)
        print(mean_absolute_error((val * self.s.to_numpy()) +
                                  self.m.to_numpy(), Y_test))
        print((val * self.s.to_numpy()) + self.m.to_numpy())
        self.model.save('combined_model')

    def test(self, fields=None):
        assert fields is not None
        self.get_labels(test_train=False)
        val = {}
        for field in t:
            print("Test data size ", len(self.X))
            model = load_model("temp_model_" + field)
            v = model.predict(self.X.to_numpy())
            mae = mean_absolute_error(self.Y[field], v)
            val[field] = mae
            # original_val = tr.Y['nFix'].to_list()
            # pred_val = np.ravel(tr.pred)[::5]
        print("Metrics is ", val)
        return val

    def __test_train_split(self):
        tr, te = train_test_split(self.df, train_size=self.ratio)
        return tr, te

    def model_process(self, fields):
        if not self.cross:
            self.cvalid = 1
        final_val = {}
        for i in range(self.cvalid):
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Processing cross validation iteration {}".format(i))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            if i == 0:
                self.train(fields, load_previous=False)
            else:
                self.train(fields, load_previous=True)
            # final_val[i] = self.test(fields)
        # print(final_val, "Total val: ", pd.DataFrame(final_val).mean(axis=1))


class PredictionCallback(tf.keras.callbacks.Callback):

    def get_labels(self, df):
        labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
        drop_labels = ['sentence_id', 'word_id', 'word',
                       'Unnamed: 0', 'otag', 'utag']

        df = df[df.columns[~df.columns.isin(drop_labels)]]
        Y = df[labels]
        X = df[df.columns[~df.columns.isin(labels)]]
        return X, Y

    def on_epoch_end(self, epoch, logs={}):
        df = pd.read_csv('temp_pos.csv')
        X_test, _ = self.get_labels(df)
        y_pred = self.model.predict(X_test.to_numpy())
        print('prediction: {} at epoch: {}'.format(y_pred, epoch))


if __name__ == '__main__':
    data = pd.read_csv('training_pos_tagged.csv')
    trial_df = pd.read_csv('trial_pos_tagged.csv')
    print("Trial df size", trial_df.shape)
    t = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
    # t = ['nFix']
    tr = Traintf(data, cross_valid=1, test_df=trial_df, split_rat=None,
                 batch_size=32, epochs=1000)
    # tr.train(fields=t)
    # tr.get_labels()
    tr.model_process(fields=t)
    # tr.train()
