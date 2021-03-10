import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error


class mae_(Callback):
    def on_epoch_end(self):
        # assert fields is not None
        self.get_labels(test_train=False)
        val = {}
        print("Test data size ", len(self.X))
        # model = load_model("temp_model_" + field)
        v = self.model.predict(self.X.to_numpy())
        mae = mean_absolute_error(self.Y[self.i], v)
    #     val[field] = mae
    #     # original_val = tr.Y['nFix'].to_list()
    #     # pred_val = np.ravel(tr.pred)[::5]
        print("Metrics is ", mae)
        return mae


def custom_mae(y_true, y_pred, s , m):
    import time
    print (type(y_pred), type(y_true))
    # print("====",y_true.to_numpy(), "====", y_pred.numpy())
    s = tf.convert_to_tensor(s.astype(np.float32))
    m = tf.convert_to_tensor(m.astype(np.float32))
    print("Tensor shape", tf.shape(y_pred),
          "true tf shape", tf.shape(y_true))
    y_res_tensor = tf.math.multiply(tf.math.add(y_pred, m), s)
    mae = keras.losses.mean_absolute_error(y_res_tensor, y_true)
    return mae