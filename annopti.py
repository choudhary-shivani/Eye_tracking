from google.colab import drive
drive.mount('/content/drive')

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
from tensorflow.python.client import device_lib
import numpy as np

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

    def get_gpu(self):
        tf.device('GPU:0')
        # device_name = tf.test.gpu_device_name()
        print(tf.config.list_physical_devices('GPU'))
        print(device_lib.list_local_devices())

    def get_labels(self, test_train=True):
        labels = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
        drop_labels = ['sentence_id', 'word_id', 'word',
                       'Unnamed: 0', 'otag', 'utag']

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

    def norma(self, x):
        return (x - x.min()) / (x.max() - x.min()), x.min().to_numpy(), \
               x.max().to_numpy()

    def train(self, fields=None, load_previous=False, 
              dropout=0.2, learning_rate=1e-4, batch=32):
        assert fields is not None
        if isinstance(fields, str):
            fields = [fields]
        val = {}
        self.get_labels()
        print("------------", dropout, learning_rate, batch)
        print ("training size ", self.X.shape)
        with tf.device('GPU:0'):
          for field in fields:
              print("Traing the NN for field - ", field)
            #   callback = tf.keras.callbacks.LearningRateScheduler(learning_rate,
            #                                                       verbose=False)
              if not load_previous:
                  field1 = field + "_" + str(dropout) + "_" + str(float(learning_rate))\
                            +  "_" + str(batch) 
                  print(field1)
                  callback = tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss",
                            factor=0.2,
                            patience=5,
                            verbose=1,
                            mode="min",
                            min_delta=0.0001)  
                  es = EarlyStopping(monitor='val_loss', mode='min', verbose=False,
                                patience=20)
                  mc = ModelCheckpoint("temp_model_" + field1,
                                  monitor='val_loss',
                                  mode='min', verbose=0, save_best_only=True)
                  print("==============Building models=============")
                  input = tf.keras.Input(shape=(self.X.to_numpy().shape[-1]),
                                        name='embed')
                  x = layers.Dense(1024, activation='relu')(input)
                  x = layers.Dropout(rate=dropout, seed=10)(x)
                  x = layers.Dense(512, activation='relu')(x)
                  x = layers.Dropout(rate=dropout, seed=10)(x)
                  x = layers.Dense(256, activation='relu')(x)
                  nfix = layers.Dropout(rate=dropout)(x)
                  nfix = layers.Dense(64, activation='relu', name='nfix0')(nfix)
                  # nfix = (layers.Dense(64, activation='relu', name='nfix0',
                  #                      kernel_regularizer='l2')(x))
                  nfix = layers.Dense(16, activation='relu', name='nfix2')(nfix)
                  nfix = layers.Dense(1, activation='relu', name='NFIX')(nfix)
                  model = Model(input, [nfix])
                  model.compile(optimizer='adam',
                                loss='mae'
                                # loss_weights={
                                #     'NFIX': 0.2,
                                #     'FFD': 0.4,
                                #     'GPT': 0.4
                                # }
                                )
                  plot_model(model, to_file='model_plot.png', show_shapes=True,
                            show_layer_names=True)
                  # print("Input data size ", len(self.X))
                  # print(self.Y[field])
                  # es = EarlyStopping(monitor='val_loss', mode='min', verbose=True,
                  #                    patience=10)
                  # mc = ModelCheckpoint('final_model_' + field, monitor='val_loss',
                  #                      mode='min', verbose=1, save_best_only=True)
                  # model.fit(self.X.to_numpy(),
                  #           self.Y[field].to_numpy(),
                  #           epochs=80,
                  #           batch_size=32,
                  #           verbose=True,
                  #           validation_split=0.2,
                  #           callbacks=[callback],
                  #           use_multiprocessing=True)
                  # model.save("temp_model_" + field)
              else:
                  print("==============Loading models=============")
                  model = load_model("temp_model_" + field)
              # model.summary()
              keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
              model.fit(self.X.to_numpy(),
                        self.Y[field].to_numpy(),
                        epochs=self.epochs,
                        batch_size=batch,
                        verbose=2,
                        validation_split=0.2,
                        callbacks=[callback, es, mc],
                        use_multiprocessing=True)

            #   model.save("temp_model_" + field)

    def test(self, fields=None, dropout=0.2, learning_rate=1e-4, batch=32):
        assert fields is not None
        self.get_labels(test_train=False)
        val = {}
        for field in fields:
            field1 = field + "_" + str(dropout) + "_" + str(float(learning_rate))\
               +  "_" + str(batch)
            print("Test data size ", len(self.X))
            model = load_model("temp_model_" + field1)
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

    def model_process(self, fields, dropout=0.2, learning_rate=1e-4, batch=32, only_acc=True):
        if not self.cross:
            self.cvalid = 1
        final_val = {}
        
        # for i in range(self.cvalid):
        #     # printm()
        #     if i == 0:
        if not only_acc:
          self.train(fields, load_previous=False, dropout=dropout, learning_rate=learning_rate, batch=batch)
            # else:
            #     self.train(fields, load_previous=True, dropout=dropout, learning_rate=learning_rate, batch=batch)
        final_val[0] = self.test(fields, dropout=dropout, learning_rate=learning_rate, batch=batch)
        print(final_val, "\n\nAverage score:\n", pd.DataFrame(final_val).mean(axis=1))
        print("Final mean {}".format(pd.DataFrame(final_val).mean(axis=1).mean()))
        return pd.DataFrame(final_val).mean(axis=1)

def optim(dropout=0.2, activation='relu', lr=1e-4, batch=32):
    os.chdir('/content/drive/MyDrive/col-competition')
    data = pd.read_csv('syllables_training_pos_tagged.csv')
    trial_df = data[:1971]
    data = data[1971:]
    data = data.fillna(0)
    trial_df = trial_df.fillna(0)
    # t = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
    t = ['nFix']
    tr = Traintf(data, cross_valid=10, test_df=trial_df, split_rat=None,
                batch_size=32, epochs=100)
    tr.model_process(fields=t, dropout=dropout, learning_rate=lr, batch=batch, only_acc=False)
    trial_df = pd.read_csv('syllables_test_pos_tagged.csv')
    tr1 = Traintf(data, cross_valid=10, test_df=trial_df, split_rat=None,
                batch_size=32, epochs=100)
    mae = tr1.model_process(fields=t, dropout=dropout, learning_rate=lr, batch=batch)
    print ("-----------------------------------------------------------------")
    return mae

os.chdir('/content/drive/MyDrive/col-competition')
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, space_eval
trials = Trials()
def opt_func(params):
    with open("optimization_ANN.txt", 'a') as f:
        f.write(str(params) + '\n')
    mae = optim(dropout=params['dropout'], lr=params['lr'], batch= params['batch'])
    with open("optimization_ANN.txt", 'a') as f:
        f.write(str(mae.to_dict()) + '\n')
    return {"loss": mae.mean(),
            "status": STATUS_OK}
with open("optimization_ANN.txt", 'w') as f:
    f.write("Started Syllables based feature - Avrage pooling  \n\n")            
trials = Trials()
space = {
    'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'lr': hp.choice('lr', [1e-4, 1e-5, 1e-3]),
    'batch' : hp.choice('batch', [16, 32, 64])
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
