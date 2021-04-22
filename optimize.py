from google.colab import drive
drive.mount('/content/drive')

!pip install GPUtil
from keras.utils.vis_utils import plot_model
import pickle
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model, activations
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection.tests.test_split import train_test_split
from sklearn.metrics import mean_absolute_error
# from custom_error import mae_
import numpy as np
from keras.optimizers import Adam

import psutil
import humanize
import os
import GPUtil as GPU
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()

class Traintf:
    def __init__(self, df, split_rat=0.1, cross_valid=10, test_df=None,
                 batch_size=32, epochs=80, context_len=10):
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
        self.context_len = context_len
        self.label = None

    def get_labels(self, test_train=True):
        labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
        drop_labels = ['word_id', 'word',
                       'Unnamed: 0', 'otag', 'utag',
                       'crossreftime', 'GPT-FFD', 'TRT-GPT', 'pps', 'phonem',
                       'sylabie']

        if test_train:
            df = self.traindf
        else:
            df = self.testdf
        df = df[df.columns[~df.columns.isin(drop_labels)]]
        self.Y = df[labels]
        self.X = df[df.columns[~df.columns.isin(labels)]]

    def lr_rate(self, epoch, lr):
        if epoch > 10:
            lr = lr * tf.math.exp(-0.05)
        elif epoch > 20:
            lr = lr * tf.abs.exp(-0.08)
        else:
            lr = 1e-4
        return lr

    def norma(self, x):
        return (x - x.min()) / (x.max() - x.min()), x.min().to_numpy(), \
               x.max().to_numpy()

    def loader(self):
        ext = 'train'
        with open("future_" + ext +".pkl", 'rb') as f:
            self.future = np.array(pickle.load(f))
        with open("history_" + ext +".pkl", 'rb') as f:
            self.past = np.array(pickle.load(f))
        with open("label_" + ext +".pkl", 'rb') as f:
            self.label = np.array(pickle.load(f))
        ext = 'test'
        with open("future_" + ext +".pkl", 'rb') as f:
            self.future_test = np.array(pickle.load(f))
        with open("history_" + ext +".pkl", 'rb') as f:
            self.past_test = np.array(pickle.load(f))
        with open("label_" + ext +".pkl", 'rb') as f:
            self.label_test = np.array(pickle.load(f))


    def train(self, fields=None, load_previous=False,
              old=None, crnt=0, dropout= 0.2, activation='relu', 
              learning_rate=1e-4, batch=32, filter=32):
        # assert fields is not None
        if isinstance(fields, str):
            fields = [fields]
        # val = {}
        field = 'CNN_ALL'
        for idx, field in enumerate(fields):
          print("Processing for field {}".format(field))
          with tf.device('GPU:0'):
              field = field + "_" + str(dropout) + "_" + str(float(learning_rate))\
               + '_' + \
               str(self.context_len)+ "_" + str(batch) +  "_" + str(filter) + "_length_memory"
              self.get_labels(test_train=False)
              X_test, Y_test = self.X, self.Y
            #   callback = tf.keras.callbacks.LearningRateScheduler(self.lr_rate,
            #                                                       verbose=False)
              callback = tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss",
                            factor=0.2,
                            patience=5,
                            verbose=1,
                            mode="min",
                            min_delta=0.0001)  
              es = EarlyStopping(monitor='val_loss', mode='min', verbose=False,
                                patience=20)
              mc = ModelCheckpoint('fm_' + field,
                                  monitor='val_loss',
                                  mode='min', verbose=0, save_best_only=True)
              if not load_previous:
                  past = Input(shape=(self.context_len, self.past[0].shape[1], 1))
                  future = Input(shape=(self.context_len, self.past[0].shape[1], 1))
                  # model A
                  x1 = layers.Conv2D(filter, (3, 3), activation=activation, padding='same')(past)

                  x2 = layers.Conv2D(filter, (5, 5), padding='same')(past)
                  x = layers.Concatenate()([x1, x2])
                  x = layers.BatchNormalization()(x)
                  x = layers.Activation(activations.relu)(x)
                  x = layers.AveragePooling2D((2, 2))(x)
                  
                  x = layers.Conv2D(filter, (3, 3), activation=activation, padding='same')(x)
                  x = layers.AveragePooling2D((2, 2))(x)
                  x = layers.AveragePooling2D((2, 2))(x)
                  x = layers.Dropout(dropout)(x)

                  # Model B
                  y = layers.Conv2D(filter, (3, 3), activation=activation, padding='same')(future)
                  y = layers.Conv2D(filter, (3, 3), activation=activation, padding='same')(y)
                  y = layers.BatchNormalization()(y)
                  y = layers.Activation(activations.relu)(y)
                  y = layers.AveragePooling2D((2, 2))(y)
                  
                  y = layers.Conv2D(filter, (3, 3), activation=activation, padding='same')(y)
                  y = layers.AveragePooling2D((2, 2))(y)
                  y = layers.AveragePooling2D((2, 2))(y)
                  y = layers.Dropout(dropout)(y)

                  common_vec = layers.Average()([layers.Flatten()(x),
                                                    layers.Flatten()(y)])
                  final = layers.Dense(2048, activation=activation, name='pre_final0_0')(
                      common_vec)
                  final = layers.Dropout(dropout)(final)
                  final = layers.Dense(1024, activation=activation, name='pre_final0_1')(
                      final)
                  final = layers.Dropout(dropout)(final)
                  final = layers.Dense(512, activation=activation, name='pre_final0_2')(
                      final)
                  final = layers.Dropout(dropout)(final)
                  final = layers.Dense(64, activation=activation, name='pre_final0_3')(
                      final)
                  # final = layers.Dropout(0.2)(final)
                  final = layers.Dense(32, activation=activation, name='pre_final0')(
                      final)
                  final = layers.Dense(16, activation=activation, name='pre_final1')(
                      final)
                  final = layers.Dense(1, name='final')(
                      final)

                  model = Model([past, future], final)
                  model.compile(optimizer=Adam(lr=learning_rate),
                                loss='mae'
                                )
                #   print(model.summary())
                  plot_model(model, to_file='model_plot_' +str(self.context_len) + '.png', 
                            show_shapes=True,
                            show_layer_names=True)
              else:
                  model = load_model('fm_'  + field)
          
              model.fit([self.past,
                        self.future], np.ravel(self.label)[3::5],
                    #   validation_split=0.2,
                      validation_data= ([self.past_test, self.future_test],
                                        np.ravel(self.label_test)[3::5]),
                      callbacks=[ es, mc],
                      verbose=2,
                      epochs=self.epochs,
                      batch_size=self.batch_size)
        # for idx, field in enumerate(fields):
        #   field = field + "_" + str(self.context_len) + "_length_memory"   
        #   model = load_model('fm_' + str(crnt) + "_" + field )       
        #   val = model.predict([self.past_test, self.future_test])
        #   # print(val, self.label_test)
        #   print ("Value of mae {} for {}".format(
        #     mean_absolute_error(np.ravel(val), np.ravel(self.label_test)[idx::5]), field))

    def test(self, dropout= 0.2, activation='relu', 
              learning_rate=1e-4, fields=None, batch=32, filter=32):
        assert fields is not None
        self.get_labels(test_train=False)
        val = {}
        x = pd.DataFrame()
        for idx, field in enumerate(fields):
            field = field + "_" + str(dropout) + "_" + str(float(learning_rate)) \
               + '_' + \
               str(self.context_len)+"_" + str(batch)+  "_" + str(filter)  + "_length_memory"
            print("Test data size ", len(self.X))
            model = load_model('fm_' + field)
            v = model.predict([self.past_test, self.future_test])
            # print(np.ravel(self.label_test)[idx::5])
            # mae = mean_absolute_error(np.ravel(self.label_test)[idx::5], v)
            # print ("Mean abs error for {}".format(field), mae)
            x[field] = np.ravel(v)
            # original_val = tr.Y['nFix'].to_list()
            # pred_val = np.ravel(tr.pred)[::5]
        return x

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
        print("Processing the dataset {} with extension {}".format((df.shape), ext))
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
                # print(pred, Y)
                X = X[X.columns[~X.columns.isin(['sentence_id', 'word_id'])]]
                vec_len = X.shape[-1]

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
        print(X.columns.to_list(),"\n\n",
        len(X.columns), "\n\n",
        pred.columns)
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

    def data_format(self, context_len=10, test= False):
        assert isinstance(context_len, int)
        # print(context_len)
        self._data_format(test=test, context_len=context_len )

def get_labels(df):
    labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
    drop_labels = ['word', 'Unnamed: 0', 'otag', 'utag',
                   'pps', 'crossreftime', 'GPT-FFD', 'TRT-GPT',
                   'phonem', 'sylabie']

    # if test_train:
    #     df = self.traindf
    # else:
    #     df = self.testdf
    df = df[df.columns[~df.columns.isin(drop_labels)]]
    Y = df[labels]
    X = df[df.columns[~df.columns.isin(labels)]]
    return X, Y

def optim(dropout=0.2, activation='relu', lr=1e-4, batch=32, filter=32):
    os.chdir('/content/drive/MyDrive/col-competition')
    data = pd.read_csv('syllables_training_pos_tagged.csv')
    # trial_df = pd.read_csv('syllables_test_pos_tagged.csv')
    trial_df = data[:1971]
    data = data[1971:]
    data = data.fillna(0)
    trial_df = trial_df.fillna(0)
    # t = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
    t = ['GPT']
    context_len=10
    print("Processing context length {}".format(context_len))
    print("======================================")
    tr = Traintf(data, cross_valid=1, test_df=trial_df, split_rat=None,
                batch_size=batch, epochs=100, context_len=context_len)
    tr.data_format(test=False, context_len=10)
    tr.data_format(test=True, context_len=10)
    tr.loader()
    tr.train(fields=t, activation='relu', dropout=dropout, learning_rate=lr,
             batch=batch, filter=filter)

    trial_df = pd.read_csv('syllables_test_pos_tagged.csv')

    # to test and generate mae
    tr1 = Traintf(data, cross_valid=1, test_df=trial_df, split_rat=None,
                batch_size=batch, epochs=75, context_len=context_len)
    # tr1.data_format(test=True, context_len=10)
    tr1.loader()
    infrence_val = tr1.test(fields=t, activation='relu', dropout=dropout, 
                            learning_rate=lr, batch=batch, filter=filter)
    final_mat = pd.concat([trial_df[['sentence_id', 'word_id',
                                     'word']], infrence_val], axis=1)
    final_mat.to_csv('inference_CNN_final.csv', index=False)
    a = pd.read_csv('truth.csv')
    b = pd.read_csv('inference_CNN_final.csv')
    mae_ = 0
    for i in t:
        x = a[i]
        y = b[i + "_" + str(dropout) + "_" + str(float(lr)) + '_' +
               str(10)+ "_" + str(batch)+  "_" + str(filter)  + "_length_memory"]
        mae = mean_absolute_error(x,y)
        mae_ += mae
        print ("{} mae is {}".format(i, mae))
    mae_ = mae_ / len(t)
    return mae_

os.chdir('/content/drive/MyDrive/col-competition')
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, space_eval
trials = Trials()
def opt_func(params):
    with open("optimization.txt", 'a') as f:
        f.write(str(params) + '\n')
    mae = optim(dropout=params['dropout'], lr=params['lr'], batch= params['batch'],
                filter= params['filter'])
    with open("optimization.txt", 'a') as f:
        f.write(str(mae) + '\n')
    return {"loss": mae,
            "status": STATUS_OK}
with open("optimization.txt", 'w') as f:
    f.write("Started Syllables based feature - Avrage pooling  \n\n")            
trials = Trials()
space = {
    'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'lr': hp.choice('lr', [1e-3, 1e-4]),
    'batch' : hp.choice('batch', [16, 32]), 
    'filter' : hp.choice('filter', [32, 64])
}

best = fmin(fn=opt_func,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
            )

print("Best: {}".format(best))
print(trials.results)
print(trials.best_trial)
print(space_eval(space, best))
