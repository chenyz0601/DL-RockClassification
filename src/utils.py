import numpy as np
import glob, os
import tensorflow as tf
from keras.callbacks import TensorBoard

def split2Tiles(path, arr, x_size=256, y_size=256, suffix='block1'):
    xx, yy = arr.shape[1], arr.shape[2]
    x_num = int(xx/x_size)
    y_num = int(yy/y_size)
    print('split into {0} tiles'.format(x_num*y_num))
    count = 0
    for i in range(x_num):
        for j in range(y_num):
            tmp = arr[:, x_size*i: x_size*(i+1), y_size*j: y_size*(j+1)]
            np.save(path+'{0}_{1}'.format(suffix, count), tmp)
            count += 1
            
def split_trn_vld_tst(path, vld_rate=0.2, tst_rate=0.1, random=True, seed=10):
    np.random.seed(seed)
    x_IDs = sorted(glob.glob(path+'X/*.npy'))
    y_IDs = sorted(glob.glob(path+'Y/*.npy'))
    if len(x_IDs) != len(y_IDs):
        raise ValueError('imgs and labels are not matched!')
    num = len(x_IDs)
    vld_num = int(num*vld_rate)
    tst_num = int(num*tst_rate)
    print('split into {0} train, {1} validation, {2} test samples'.format(num-vld_num-tst_num, vld_num, tst_num))
    idx = np.arange(num)
    if random:
        np.random.shuffle(idx)
    X_tst = [x_IDs[k] for k in idx[:tst_num]]
    X_vld = [x_IDs[k] for k in idx[tst_num:tst_num+vld_num]]
    X_trn = [x_IDs[k] for k in idx[tst_num+vld_num:]]
    Y_tst = [y_IDs[k] for k in idx[:tst_num]]
    Y_vld = [y_IDs[k] for k in idx[tst_num:tst_num+vld_num]]
    Y_trn = [y_IDs[k] for k in idx[tst_num+vld_num:]]
    return X_trn, Y_trn, X_vld, Y_vld, X_tst, Y_tst

def remove_unknown_rocks(path, rate, w=256, h=256):
    print('removing tiles with more than {0} unknown rock'.format(rate))
    x_IDs = sorted(glob.glob(path+'X/*.npy'))
    y_IDs = sorted(glob.glob(path+'Y/*.npy'))
    if len(x_IDs) != len(y_IDs):
        raise ValueError('imgs and labels are not matched!')
    num = len(x_IDs)
    th = int(w*h*rate)
    for i in range(num):
        tmp = np.load(y_IDs[i])[1,:,:]
        if tmp.sum() > th:
            os.remove(y_IDs[i])
            os.remove(x_IDs[i])
                
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', hist_freq=0, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, histogram_freq=hist_freq, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()