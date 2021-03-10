from keras.utils.vis_utils import plot_model
import pickle
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection.tests.test_split import train_test_split
from sklearn.metrics import mean_absolute_error
from custom_error import mae_
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.device('cpu:0')


def get_labels(df):
    labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
    drop_labels = ['word', 'Unnamed: 0', 'otag', 'utag',
                   'pps', 'crossreftime', 'GPT-FFD', 'TRT-GPT']

    # if test_train:
    #     df = self.traindf
    # else:
    #     df = self.testdf
    df = df[df.columns[~df.columns.isin(drop_labels)]]
    Y = df[labels]
    X = df[df.columns[~df.columns.isin(labels)]]
    return X, Y


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
        self.future = None
        self.past = None
        self.label = None

    def get_labels(self, test_train=True):
        labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
        drop_labels = ['word',
                       'Unnamed: 0', 'otag', 'utag',
                       'crossreftime', 'GPT-FFD', 'TRT-GPT', 'pps']

        if test_train:
            df = self.traindf
        else:
            df = self.testdf
        df = df[df.columns[~df.columns.isin(drop_labels)]]
        if test_train:
            self.traindf = df
        else:
            self.testdf = df

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

    def loader(self):
        ext = 'train'
        with open("future_" + ext + ".pkl", 'rb') as f:
            self.future = np.array(pickle.load(f))
        with open("history_" + ext + ".pkl", 'rb') as f:
            self.past = np.array(pickle.load(f))
        with open("label_" + ext + ".pkl", 'rb') as f:
            self.label = np.array(pickle.load(f))
        ext = 'test'
        with open("future_" + ext + ".pkl", 'rb') as f:
            self.future_test = np.array(pickle.load(f))
        with open("history_" + ext + ".pkl", 'rb') as f:
            self.past_test = np.array(pickle.load(f))
        with open("label_" + ext + ".pkl", 'rb') as f:
            self.label_test = np.array(pickle.load(f))

    def train(self, fields=None, load_previous=False,
              old=None, crnt=0):
        # assert fields is not None
        if isinstance(fields, str):
            fields = [fields]
        # val = {}
        field = 'CNN_ALL'
        self.get_labels(test_train=False)
        X_test, Y_test = self.X, self.Y
        callback = tf.keras.callbacks.LearningRateScheduler(self.lr_rate,
                                                            verbose=False)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=False,
                           patience=30)
        mc = ModelCheckpoint('fm_' + str(crnt) + "_" + field,
                             monitor='val_loss',
                             mode='min', verbose=0, save_best_only=True)

        past = Input(shape=(10, self.past[0].shape[1], 1))
        future = Input(shape=(10, self.past[0].shape[1], 1))
        # model A
        x = layers.Conv2D(32, (3, 3), activation='relu')(past)
        x = layers.MaxPool2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPool2D((2, 2))(x)
        # x = layers.Conv2D(64, (3, 3), activation='relu')(x)

        # Model B
        y = layers.Conv2D(32, (3, 3), activation='relu')(future)
        y = layers.MaxPool2D((2, 2))(y)
        y = layers.Conv2D(64, (3, 3), activation='relu')(y)
        y = layers.MaxPool2D((2, 2))(y)
        # y = layers.Conv2D(64, (3, 3), activation='relu')(y)

        common_vec = layers.Concatenate()([layers.Flatten()(x),
                                           layers.Flatten()(y)])
        final = layers.Dense(2048, activation='relu', name='pre_final0_0')(
            common_vec)
        final = layers.Dense(1024, activation='relu', name='pre_final0_1')(
            final)
        final = layers.Dense(512, activation='relu', name='pre_final0_2')(
            final)
        final = layers.Dense(64, activation='relu', name='pre_final0')(
            final)
        final = layers.Dropout(0.2)(final)
        final = layers.Dense(16, activation='relu', name='pre_final1')(
            final)
        final = layers.Dense(5, name='final')(
            final)

        model = Model([past, future], final)
        model.compile(optimizer='adam',
                      loss='mae'
                      )
        print(model.summary())
        plot_model(model, to_file='model_plot.png', show_shapes=True,
                   show_layer_names=True)
        model.fit([self.past, self.future],
                  self.label,
                  validation_split=0.2,
                  callbacks=[callback, es, mc],
                  verbose=True,
                  epochs=self.epochs,
                  batch_size=self.batch_size)
        val = model.predict([self.past_test, self.future_test])
        print(val)

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
                self.train(fields, load_previous=False, old=None, crnt=i)
            else:
                self.train(fields, load_previous=True, old=i - 1, crnt=i)
            # final_val[i] = self.test(fields)
        print(final_val, "Total val: ", pd.DataFrame(final_val).mean(axis=1))

    def _data_format(self, context_len=10, test=False):
        if test:
            self.get_labels(test_train=False)
            df = self.testdf
            ext = 'test'
        else:
            self.get_labels(test_train=True)
            df = self.traindf
            ext = 'train'

        all_history = []
        all_future = []
        all_label = []
        labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
        for i in np.unique(df['sentence_id'].to_list()):
            # print(self.traindf[self.traindf['sentence_id'] == i])
            for j, word in enumerate(np.unique(df['word_id'][df[
                                     'sentence_id'] == i])):
                X, Y = get_labels(df[df['sentence_id'] == i])
                pred = Y[X['word_id'] == word][labels]
                X = X[X.columns[~X.columns.isin(['sentence_id'])]]
                vec_len = X.shape[-1]
                # print(X.columns.to_list(),
                #       pred.columns)
                np_x = X.to_numpy()
                X_past = np.zeros(vec_len *
                                  context_len).reshape(-1, vec_len)
                X_fut = np.zeros(vec_len *
                                 context_len).reshape(-1, vec_len)
                if context_len > j:
                    update_idx = context_len - j
                else:
                    update_idx = 1
                if context_len > j:
                    copy_idx = j
                else:
                    copy_idx = context_len - 1
                # print(i, j, update_idx, copy_idx)
                X_past[update_idx - 1:] = np_x[j - copy_idx:j + 1]
                # if context_len > j:
                #     copy_idx = j
                # else:
                #     copy_idx = context_len - 1
                # print(update_idx, copy_idx, j)
                # print("=====================")
                # print(np_x)

                X_fut[:len(np_x[j: j + context_len])] = np_x[j: j + context_len]
                # print(X_fut)
                # time.sleep(2)
                # np_x = np.concatenate([X_past, np_x, X_past], axis=0)
                all_label.append(pred)
                all_future.append(X_fut)
                all_history.append(X_past)
        print([len(i) for i in [all_history, all_label, all_future]])
        if test:
            self.past_test = all_history
            self.future_test = all_future
            self.label_test = all_label
            with open("future_" + ext + ".pkl", 'wb') as f:
                pickle.dump(all_future, f)
            with open("history_" + ext + ".pkl", 'wb') as f:
                pickle.dump(all_history, f)
            with open("label_" + ext + ".pkl", 'wb') as f:
                pickle.dump(all_label, f)
        else:
            self.past = all_history
            self.future = all_future
            self.label = all_label
            with open("future_" + ext + ".pkl", 'wb') as f:
                pickle.dump(all_future, f)
            with open("history_" + ext + ".pkl", 'wb') as f:
                pickle.dump(all_history, f)
            with open("label_" + ext + ".pkl", 'wb') as f:
                pickle.dump(all_label, f)

    def data_format(self, context_len=10, test=False):
        assert isinstance(context_len, int)
        self._data_format(test=test)


if __name__ == '__main__':
    data = pd.read_csv('train_lit_val.csv')
    trial_df = pd.read_csv('trial_lit_val.csv')
    data = data.fillna(0)
    trial_df = trial_df.fillna(0)
    t = ['nFix']
    tr = Traintf(data, cross_valid=1, test_df=trial_df, split_rat=None,
                 batch_size=32, epochs=2)
    tr.data_format()
    tr.data_format(test=True)
    tr.loader()
    tr.train()
