import glob
import re
import keras
from keras.optimizers import adam
from keras.layers import Input
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.losses import categorical_crossentropy
import os, random
from .networks import SegmentorNet, AdversarialNet
from .utils import TrainValTensorBoard

class AdvSeg:
    
    def __init__(self, num_bands=10,
                 dim_width=256,
                 dim_height=256,
                 num_labels=10):
        self.num_bands = num_bands
        self.dim_width = dim_width
        self.dim_height = dim_height
        self.num_labels = num_labels
        self.model = None
        self.model_type = None
        self.callbackList = None
        self.init_epoch = 0
    
    def build_SegmentorNet(self, k_size = (3, 3),
                           n_ch_list=[64, 64, 64, 64],
                           k_init='lecun_normal',
                           activation='selu'):
        inp = Input((self.dim_width, self.dim_height, self.num_bands))
        outp = SegmentorNet(inp, self.num_labels, n_ch_list, k_size, k_init, activation)
        self.model = Model(inputs=[inp], outputs=[outp])
        self.model_type = 'Segmentor'
    
    def build_AdvSegNet(self, k_size = (3, 3),
                        seg_ch_list=[64, 64, 64, 64],
                        adv_ch_list=[64, 64, 64],
                        br_ch=64,
                        k_init='lecun_normal',
                        activation='selu'):
        img_inp = Input((self.dim_width, self.dim_height, self.num_bands))
        label_inp = Input((self.dim_width, self.dim_height, self.num_labels))
        pred_inp = SegmentorNet(img_inp, self.num_labels, seg_ch_list, k_size, k_init, activation)
        
        # with K.variable_scope("AdversarialNet", reuse=True):
        adv_model = Model(inputs=[img_inp, label_inp],
                          outputs=AdversarialNet(img_inp, label_inp, adv_ch_list,
                                                 k_size, k_init, activation, br_ch))
        self.adv_out_true = adv_model([img_inp, label_inp])
        self.adv_out_fake = adv_model([img_inp, pred_inp])
        
        self.model = Model(inputs=[img_inp, label_inp],
                           outputs=[pred_inp])
        self.model_type = 'AdvSeg'
        
    def compile_model(self, scale=1e-2, lr=1e-3, verbose=True):
        optimizer = adam(lr=lr)
        if self.model_type == 'Segmentor':
            print('compiling Segmentor only ...')
            loss = 'categorical_crossentropy'
            # build the whole computational graph with model, loss and optimizer
            # 'accuracy' is defaultly categorical_accuracy
            self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            # print parameters of each layer
            if verbose:
                print(self.model.summary())
        elif self.model_type == 'AdvSeg':
            print('compiling Segmentor with Adversarial net ...')
            self.model.compile(loss=self.AdvSegLoss, optimizer=optimizer, metrics=['accuracy'])
            # print parameters of each layer
            if verbose:
                print(self.model.summary())
        else:
            raise ValueError('no model to be compiled!')
                    
    def fit_model_generator(self, train_generator,
                            valid_generator,
                            verbose=1,
                            workers=1,
                            use_multiprocessing=False,
                            use_tfboard=True,
                            num_epochs=10):
            self.build_callbackList(use_tfboard)
            print('fitting model {0}'.format(self.model_type))
            self.model.fit_generator(generator=train_generator,
                                     validation_data=valid_generator,
                                     verbose=verbose,
                                     epochs=self.init_epoch+num_epochs,
                                     callbacks=self.callbackList,
                                     workers=workers,
                                     use_multiprocessing=use_multiprocessing,
                                     initial_epoch=self.init_epoch)
    
    def AdvSegLoss(self, y_true, y_pred, scale=1e-1):
        mce = K.mean(K.mean(categorical_crossentropy(y_true, y_pred), axis=-1), axis=-1)
        bce_true = K.log(self.adv_out_true)
        bce_fake = K.log(1. - self.adv_out_fake)
        return mce-scale*(bce_true+bce_fake)
    
    def build_callbackList(self, use_tfboard, log_dir='./logs'):        
        if self.model_type == None:
            raise ValueError('model is not built yet, please build 1D, 2D or 3D convnet model')
        else:
            path = './{0}'.format(self.model_type)

        # Model Checkpoints
        if not os.path.exists(path):
            os.makedirs(path)
        filepath=path+'/weights-{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max')

        # Bring all the callbacks together into a python list
        self.callbackList = [checkpoint]
                    
        # Tensorboard
        if use_tfboard:
            path = log_dir+'/{0}'.format(self.model_type)
            tensorboard = TrainValTensorBoard(log_dir=path)
            self.callbackList.append(tensorboard)
        
    def load_checkpoint(self):

        if self.model_type == None:
            raise ValueError('model is not built yet, please build Segmentor or AdvSeg!')
        else:
            path = './{0}'.format(self.model_type)
        try:
            checkfile = sorted(glob.glob(path+"/weights-*-*.hdf5"))[-1]
            self.model.load_weights(checkfile)
            self.init_epoch = int(re.search(r"weights-(\d*)-", checkfile).group(1))
            print("{0} weights loaded, resuming from epoch {1}".format(self.model_type, self.init_epoch))
        except IndexError:
            try:
                self.model.load_weights(path+"/model-weights.hdf5")
                print("{0} weights loaded, starting from epoch {1}".format(self.model_type, self.init_epoch))
            except OSError:
                pass

    def fit_model(self, X_trn,
                  Y_trn,
                  verbose=1,
                  validation_split=0.2,
                  batch_size=6,
                  num_epochs=10):
        self.model.fit(x=X_trn,
                       y=Y_trn,
                       verbose=verbose,
                       validation_split=validation_split, 
                       batch_size=batch_size,
                       epochs=self.init_epoch+num_epochs,
                       callbacks=self.callbackList,
                       initial_epoch=self.init_epoch)

    def save_weights(self, suffix='model-1'):

        if self.model_type == None:
            raise ValueError('model is not built yet, please build 1D, 2D or 3D convnet model')
        else:
            path = './{0}'.format(self.model_type)

        filepath = path+'/{0}.hdf5'.format(suffix)
        self.model.save_weights(filepath=filepath)
        return
        
    def load_weights(self, filepath):

        if self.model_type == None:
            raise ValueError('model is not built yet, please build 1D, 2D or 3D convnet model')

        self.model.load_weights(filepath=filepath)
        return
        
    def predict(self, X_tst, verbose=1):
        
        return self.model.predict(X_tst, verbose=verbose)  