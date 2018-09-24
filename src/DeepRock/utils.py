import numpy as np
import glob, os
import matplotlib.pyplot as plt

def make_trainable(net, val):
    # net.trainable = val
    for l in net.layers:
        l.trainable = val

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
            
def remove_boundary(path):
    print('removing tiles with bad boundary')
    x_IDs = sorted(glob.glob(path+'X/*.npy'))
    y_IDs = sorted(glob.glob(path+'Y/*.npy'))
    if len(x_IDs) != len(y_IDs):
        raise ValueError('imgs and labels are not matched!')
    num = len(x_IDs)
    for i in range(num):
        tmp = np.load(x_IDs[i])
        if np.min(tmp) == 0.0:
            os.remove(y_IDs[i])
            os.remove(x_IDs[i])
            
def get_acc_cls(tst_true, preds):
    acc_cls = np.zeros(tst_true.shape[-1])
    for idx in range(tst_true.shape[0]):
        tmp_pred = np.argmax(preds[idx,:,:,:], axis=2)
        tmp_tst = np.argmax(tst_true[idx,:,:,:], axis=2)
        for cls in range(tst_true.shape[-1]):
            tmp_pred_ = np.copy(tmp_pred)
            tmp_tst_ = np.copy(tmp_tst)
            tmp_pred_[np.where(tmp_pred_ != cls)] = -1
            tmp_tst_[np.where(tmp_tst_ != cls)] = -1
            acc_cls[cls] += np.where(tmp_pred_ == tmp_tst_)[0].shape[0]/(256**2)
    return acc_cls/tst_true.shape[0]

def get_acc(preds, tst_true):
    acc_list = []
    for i in range(tst_true.shape[0]):
        tmp_pred = np.argmax(preds[i,:,:,:], axis=2)
        tmp_tst = np.argmax(tst_true[i,:,:,:], axis=2)
        acc_list.append(np.where(tmp_pred == tmp_tst)[0].shape[0]/(256*256))
    print('mean accuracy on test data is {0}, std is {1}'.format(np.mean(acc_list), np.std(acc_list)))
    print('max is {0}, min is {1}'.format(max(acc_list), min(acc_list)))
    
def plot_pg(idx, preds, tst_true):
    _pred = np.argmax(preds[idx,:,:,:], axis=2)
    _tst = np.argmax(tst_true[idx,:,:,:], axis=2)
    plt.subplot(221)
    plt.imshow(_tst)
    plt.colorbar()
    plt.title('ground truth')
    plt.subplot(222)
    plt.imshow(_pred)
    plt.colorbar()
    plt.title('prediction')
    plt.show()
    
def test_fn(model, data, plot_from=0, plot_end=None, verbose=0):
    """
    model: keras model with trained weights
    data: instance of DataGenerator
    plot_from, plot_end: default value will plot all test images, 
                         set plot_end to 0 will not plot any image
    return: an array of accuracy of each class
    """
    # get test data X and labels y
    print('loading test data ...')
    X = data.getitem(0)[0]
    y = data.getitem(0)[1]
    
    # predict
    print('predicting with {0}...'.format(data.dtype))
    preds = model.predict(X, verbose=verbose)
    
    # compute the accuracy
    get_acc(preds, y)
    
    # plot
    if plot_end is None:
        plot_end = data.batch_size
    for i in range(plot_from, plot_end):
        print(i)
        plot_pg(i, preds, y)
    
    # return accuracy for each class
    return get_acc_cls(y, preds)