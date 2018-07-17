import glob
import re
import keras
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Input
from keras.models import Model
import os, random
from .networks import SegmentationNet, AdversarialNet
from .utils import TrainValTensorBoard

class AdvSeg:
    
    def __init__(self, dtype='sent',
                 dim_width=256,
                 dim_height=256,
                 num_labels=10):
        if dtype == 'sent':
            self.dtype = dtype
            self.num_bands = 10
        elif dtype == 'sent_geo':
            self.dtype = dtype
            self.num_bands = 23
        else:
            raise ValueError('unknown dtype, should be sent or sent_geo!')
        self.dim_width = dim_width
        self.dim_height = dim_height
        self.num_labels = num_labels
        self.model = None
        self.model_type = None
        self.callbackList = None
        self.scale = None
        self.init_epoch = 0
        self.img_shape = (dim_width, dim_height, self.num_bands)
        self.label_shape = (dim_width, dim_height, num_labels)
    
    def build_SegmentationNet(self, k_size = (3, 3),
                           n_ch_list=[64, 64, 64, 64],
                           k_init='lecun_normal',
                           activation='selu'):
        self.model = SegmentationNet(self.dim_width, self.dim_height,
                                     self.num_bands, self.num_labels,
                                     n_ch_list, k_size, k_init, activation)
        self.model_type = 'Segmentation'
    
    def build_AdvSegNet(self, k_size = (3, 3),
                        seg_ch_list=[64, 64, 64, 64],
                        adv_ch_list=[64, 64, 64],
                        br_ch=64,
                        k_init='lecun_normal',
                        activation='selu'):
        self.seg_model = SegmentationNet(self.dim_width, self.dim_height,
                                         self.num_bands, self.num_labels,
                                         seg_ch_list, k_size, k_init, activation)
        img_inp = Input(self.img_shape)
        label_inp = Input(self.label_shape)
        pred_inp = self.seg_model(img_inp)
        
        # with K.variable_scope("AdversarialNet", reuse=True):
        self.adv_model = AdversarialNet(self.dim_width, self.dim_height,
                                        self.num_bands, self.num_labels, adv_ch_list,
                                        k_size, k_init, activation, br_ch)
        self.adv_out_true = self.adv_model([img_inp, label_inp])
        self.adv_out_fake = self.adv_model([img_inp, pred_inp])
        
        # self.model = Model(inputs=[img_inp, label_inp],
        #                    outputs=[pred_inp])
        self.model_type = 'AdvSeg'
        
    def compile_model(self, scale=1e-1, lr=1e-3, verbose=True):
        print('compiling adam optimizer with learning rate {0}'.format(lr))
        if self.model_type == 'Segmentation':
            print('compiling Segmentation only ...')
            optimizer = adam(lr=lr)
            loss = 'categorical_crossentropy'
            # build the whole computational graph with model, loss and optimizer
            # 'accuracy' is defaultly categorical_accuracy
            self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            # print parameters of each layer
            if verbose:
                print(self.model.summary())
        elif self.model_type == 'AdvSeg':
            self.scale = scale
            seg_opt = adam(lr=lr)
            adv_opt = adam(lr=lr)
            
            # compile
            self.seg_model.compile(loss=self.SegLoss, optimizer=seg_opt)
            self.adv_model.compile(loss=self.AdvLoss, optimizer=adv_opt)
            
            print('compiling Segmentation + Adversarial net with lambda {0} ...'.format(self.scale))
            # self.model.compile(loss=loss, optimizer=seg_adv_opt, metrics=['accuracy'])
            # print parameters of each layer
            if verbose:
                print(self.seg_model.summary())
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
            if self.model_type == 'Segmentation':
                self.model.fit_generator(generator=train_generator,
                                         validation_data=valid_generator,
                                         verbose=verbose,
                                         epochs=self.init_epoch+num_epochs,
                                         callbacks=self.callbackList,
                                         workers=workers,
                                         use_multiprocessing=use_multiprocessing,
                                         initial_epoch=self.init_epoch)
            elif self.model_type == 'AdvSeg':
                for epoch in range(self.init_epoch, self.init_epoch+num_epochs):
                    print('training Segmentation model ...')
                    self.seg_model.fit_generator(generator=train_generator,
                                             validation_data=valid_generator,
                                             verbose=verbose,
                                             epochs=epoch+1,
                                             callbacks=self.callbackList,
                                             workers=workers,
                                             use_multiprocessing=use_multiprocessing,
                                             initial_epoch=epoch)
                    print('training Adversarial model ...')
                    self.adv_model.fit_generator(generator=train_generator,
                                             verbose=0,
                                             epochs=epoch+1,
                                             callbacks=None,
                                             workers=workers,
                                             use_multiprocessing=use_multiprocessing,
                                             initial_epoch=epoch)
    
    def SegLoss(self, y_true, y_pred):
        mce = K.mean(K.mean(categorical_crossentropy(y_true, y_pred), axis=-1), axis=-1)
        bce_fake = - K.log(self.adv_out_fake)
        return mce + self.scale * bce_fake
    
    def AdvLoss(self, y_true, y_pred):
        bce_true = - K.log(self.adv_out_true)
        bce_fake = - K.log(1. - self.adv_out_fake)
        return bce_true + bce_fake
    
    def build_callbackList(self, use_tfboard=True):        
        if self.model_type == None:
            raise ValueError('model is not built yet, please build Segmentation or AdvSeg')
        else:
            path = './{0}/{1}'.format(self.model_type, self.dtype)

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
            tfpath = './logs/{0}/{1}'.format(self.model_type, self.dtype)
            tensorboard = TrainValTensorBoard(log_dir=tfpath)
            self.callbackList.append(tensorboard)
        
    def load_checkpoint(self):

        if self.model_type == None:
            raise ValueError('model is not built yet, please build Segmentation or AdvSeg!')
        else:
            path = './{0}/{1}'.format(self.model_type, self.dtype)
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