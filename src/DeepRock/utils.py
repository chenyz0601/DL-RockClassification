import numpy as np
import glob, os
import matplotlib.pyplot as plt

def get_data(Xids, Yids, num, l=100):
    """
    Xids: id of X
    Yids: id of Y
    num: number of samples
    l: number of different tiles
    return: matirx in shape (num, feats) and (num)
    """
    if len(Xids) != len(Yids):
        raise ValueError('X and Y are not match!!')
    outX = np.zeros([num, 29])
    outY = np.zeros(num)
    count = 0
    for _ in range(l):
        idx = np.random.randint(0, len(Xids)-1)
        xx = np.load(Xids[idx])
        yy = np.argmax(np.load(Yids[idx]), axis=0)
        for _ in range(int(num/l)):
            x = np.random.randint(0,255)
            y = np.random.randint(0,255)
            outX[count,:] = xx[:,x,y]
            outY[count] = yy[x,y]
            count += 1
    return outX, outY

def get_XY(X_IDs_temp, Y_IDs_temp, dtype):
	'Generates data containing batch_size samples' 
	# X_out : (n_samples, *dim, n_channels)
	# Y_out : (n_samples, *dim, n_classes)
	# Initialization
	X_out = []
	Y_out = []
	if dtype == 'sent_ast_geo':
		for i in range(len(X_IDs_temp)):
			X_out.append(np.transpose(np.load(X_IDs_temp[i]), [1,2,0]))
			Y_out.append(np.transpose(np.load(Y_IDs_temp[i]), [1,2,0]))
	elif dtype == 'sent':
		for i in range(len(X_IDs_temp)):
			X_out.append(np.transpose(np.load(X_IDs_temp[i])[:10,:,:], [1,2,0]))
			Y_out.append(np.transpose(np.load(Y_IDs_temp[i]), [1,2,0]))
	elif dtype == 'ast':
		for i in range(len(X_IDs_temp)):
			X_out.append(np.transpose(np.load(X_IDs_temp[i])[10:16,:,:], [1,2,0]))
			Y_out.append(np.transpose(np.load(Y_IDs_temp[i]), [1,2,0]))
	elif dtype == 'sent_ast':
		for i in range(len(X_IDs_temp)):
			X_out.append(np.transpose(np.load(X_IDs_temp[i])[:16,:,:], [1,2,0]))
			Y_out.append(np.transpose(np.load(Y_IDs_temp[i]), [1,2,0]))
	else:
		raise ValueError('unknown dtype, should be sent_geo or sent')
	return np.asarray(X_out), np.asarray(Y_out)

def make_trainable(net, val):
    # net.trainable = val
    for l in net.layers:
        l.trainable = val

def split2Tiles(path, arr, x_size=256, y_size=256, suffix='block1', max_num=float('inf')):
    xx, yy = arr.shape[1], arr.shape[2]
    x_num = int(xx/x_size)
    y_num = int(yy/y_size)
    print('split into {0} tiles'.format(x_num*y_num))
    count = 0
    for i in range(x_num):
        for j in range(y_num):
            if count >= max_num:
                break
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
        tmp = np.load(x_IDs[i])[0,:,:]
        if tmp.min() == 0.0:
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

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
