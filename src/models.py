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
from .utils import TrainValTensorBoard, make_trainable

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
        self.seg_model = None
        self.adv_model = None
        self.adv_seg_model = None
        self.model_type = None
        self.callbackList = None
        self.init_epoch = 0
        self.img_shape = (dim_width, dim_height, self.num_bands)
        self.label_shape = (dim_width, dim_height, num_labels)
    
    def build_SegmentationNet(self, k_size = (3, 3),
                              n_ch_list=[64, 64, 64, 64],
                              k_init='lecun_normal',
                              activation='selu', 
                              lr=1e-3, 
                              verbose=False):
        img_inp = Input(self.img_shape, name='image_input')
        self.model_type = 'Segmentation'
        opt = adam(lr=lr)
        
        # build the segmentation model
        self.seg_model = SegmentationNet(img_inp,
                                         self.num_labels,
                                         n_ch_list,
                                         k_size,
                                         k_init,
                                         activation)
        print('compiling Segmentation only, lr is {0} ...'.format(lr))
        self.seg_model.compile(opt, 
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        if verbose:
            print('summary of adversarial net:')
            print(self.seg_model.summary())
    
    def build_AdvSegNet(self, k_size = (3, 3),
                        seg_ch_list=[64, 64, 64, 64],
                        adv_ch_list=[64, 64, 64],
                        br_ch=64,
                        k_init='lecun_normal',
                        activation='selu', 
                        scale=1e-1, 
                        seg_lr=1e-3, 
                        adv_lr=1e-3, 
                        verbose=False):
        img_inp = Input(self.img_shape, name='image_input')
        label_inp = Input(self.label_shape, name='label_input')
        self.model_type = 'AdvSeg'
        adv_opt = adam(lr=adv_lr)
        seg_opt = adam(lr=seg_lr)
        
        # build up adversarial model
        self.adv_model = AdversarialNet(img_inp,
                                        label_inp,
                                        adv_ch_list,
                                        k_size,
                                        k_init,
                                        activation,
                                        br_ch)
                
        # make adversarial model not trainable and create the freezed adv_model
        make_trainable(self.adv_model, False)
        adv_freeze = Model(inputs=self.adv_model.inputs, outputs=self.adv_model.outputs)
        adv_freeze.compile(adv_opt, 
                               loss='binary_crossentropy',  
                               metrics=['accuracy'])
        
        # build up segmentation model
        self.seg_model = SegmentationNet(img_inp,
                                         self.num_labels,
                                         seg_ch_list,
                                         k_size,
                                         k_init,
                                         activation)
        
        # compile segmentation model
        self.seg_model.compile(seg_opt,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        if verbose:
            print('summary of segmentation net:')
            print(self.seg_model.summary())
        
        # get the prediction of seg_model 
        pred = self.seg_model(img_inp)
        
        # input pred into adv_freeze to get prob
        prob = adv_freeze([img_inp, pred])
        
        # stack seg and adv model
        self.adv_seg_model = Model(inputs=[img_inp, label_inp], outputs=[pred, prob])
        
        # compile stacked seg and adv model
        self.adv_seg_model.compile(seg_opt, 
                                   loss=['categorical_crossentropy',
                                         'binary_crossentropy'], 
                                   loss_weights=[1., scale], 
                                   metrics=['accuracy', 'accuracy'])
        if verbose:
            print('summary of adv+seg net:')
            print(self.adv_seg_model.summary())
        
        # compile adversarial model
        make_trainable(self.adv_model, True)
        self.adv_model.compile(adv_opt, 
                               loss='binary_crossentropy',  
                               metrics=['accuracy'])            
        if verbose:
            print('summary of adversarial net:')
            print(self.adv_model.summary())
                    
    def fit_model_generator(self, train_generator,
                            valid_generator,
                            verbose=1,
                            workers=1,
                            use_multiprocessing=False,
                            use_tfboard=True,
                            num_epochs=10,
                            alt_num=10):
            print('fitting model {0}'.format(self.model_type))
            if self.model_type == 'Segmentation':
                self.build_callbackList(use_tfboard, 'val_acc')
                self.seg_model.fit_generator(generator=train_generator,
                                             validation_data=valid_generator,
                                             verbose=verbose,
                                             epochs=self.init_epoch+num_epochs,
                                             callbacks=self.callbackList,
                                             workers=workers,
                                             use_multiprocessing=use_multiprocessing,
                                             initial_epoch=self.init_epoch)
            elif self.model_type == 'AdvSeg':
                self.build_callbackList(use_tfboard, 'val_model_2_acc')
                for i in range(alt_num):
                    print('round {0} fitting adv_model'.format(i))
                    train_generator.phase = 'AdversarialNet'
                    valid_generator.phase = 'AdversarialNet'
                    self.adv_model.fit_generator(generator=train_generator,
                                                 validation_data=valid_generator,
                                                 verbose=verbose,
                                                 epochs=self.init_epoch+num_epochs,
                                                 callbacks=self.callbackList[-1],
                                                 workers=workers,
                                                 use_multiprocessing=use_multiprocessing,
                                                 initial_epoch=self.init_epoch)
                    print('round {0} fitting seg_model'.format(i))
                    train_generator.phase = 'SegmentationNet'
                    valid_generator.phase = 'SegmentationNet'
                    self.adv_seg_model.fit_generator(generator=train_generator,
                                                     validation_data=valid_generator,
                                                     verbose=verbose,
                                                     epochs=self.init_epoch+num_epochs,
                                                     callbacks=self.callbackList,
                                                     workers=workers,
                                                     use_multiprocessing=
                                                     use_multiprocessing,
                                                     initial_epoch=self.init_epoch)
    
    def build_callbackList(self, use_tfboard=True, monitor='val_acc'):
        if self.model_type == None:
            raise ValueError('model is not built yet, please build Segmentation or AdvSeg')
        else:
            path = './{0}/{1}'.format(self.model_type, self.dtype)

        # Model Checkpoints
        if not os.path.exists(path):
            os.makedirs(path)
        filepath=path+'/weights-{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(filepath,
                                     monitor=monitor,
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
        self.adv_seg_model.save_weights(filepath=filepath)
        return
        
    def load_weights(self, filepath):

        if self.model_type == 'Segmentation':
            self.seg_model.load_weights(filepath=filepath)
        elif self.model_type == 'AdvSeg':
            self.adv_seg_model.load_weights(filepath=filepath)
        else:
            raise ValueError('model is not built yet')
        
    def predict(self, X_tst, verbose=0):
        return self.seg_model.predict(X_tst, verbose=verbose)  