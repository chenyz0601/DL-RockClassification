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
from .utils import make_trainable
import tensorflow as tf
from keras.callbacks import TensorBoard

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
                                        br_ch,
                                        'adv_model')
                
        # make adversarial model not trainable and create the freezed adv_model
        make_trainable(self.adv_model, False)
        adv_freeze = Model(inputs=self.adv_model.inputs,
                           outputs=self.adv_model.outputs,
                           name='adv_model_freeze')
        adv_freeze.compile(adv_opt, 
                           loss='binary_crossentropy',  
                           metrics=['accuracy'])
        
        # build up segmentation model
        self.seg_model = SegmentationNet(img_inp,
                                         self.num_labels,
                                         seg_ch_list,
                                         k_size,
                                         k_init,
                                         activation, 
                                         'seg_model')
        
        # compile segmentation model
        self.seg_model.compile(seg_opt,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        if verbose:
            print('summary of {0}:'.format(self.seg_model.name))
            print(self.seg_model.summary())
        
        # get the prediction of seg_model 
        pred = self.seg_model(img_inp)
        
        # input pred into adv_freeze to get prob
        prob = adv_freeze([img_inp, pred])
        
        # stack seg and adv model
        self.adv_seg_model = Model(inputs=[img_inp, label_inp],
                                   outputs=[pred, prob], 
                                   name='adv_seg_model')
        
        # compile stacked seg and adv model
        self.adv_seg_model.compile(seg_opt, 
                                   loss=['categorical_crossentropy',
                                         'binary_crossentropy'], 
                                   loss_weights=[1., scale], 
                                   metrics=['accuracy'])
        if verbose:
            print('summary of {0}:'.format(self.adv_seg_model.name))
            print(self.adv_seg_model.summary())
        
        # compile adversarial model
        make_trainable(self.adv_model, True)
        self.adv_model.compile(adv_opt, 
                               loss='binary_crossentropy',  
                               metrics=['accuracy'])            
        if verbose:
            print('summary of {0}:'.format(self.adv_model.name))
            print(self.adv_model.summary())
                    
    def fit_model_generator(self, train_generator,
                            valid_generator,
                            verbose=1,
                            workers=1,
                            use_multiprocessing=False,
                            use_tfboard=True,
                            adv_epochs=10,
                            adv_steps_per_epoch=10,
                            seg_epochs=10,
                            seg_steps_per_epoch=10,
                            num_round=1):
            print('fitting model {0}'.format(self.model_type))
            if self.model_type == 'Segmentation':
                cl = self.build_callbackList(use_tfboard, 'val_acc')
                self.seg_model.fit_generator(generator=train_generator,
                                             steps_per_epoch=seg_steps_per_epoch,
                                             validation_data=valid_generator,
                                             verbose=verbose,
                                             epochs=self.init_epoch+seg_epochs,
                                             callbacks=cl,
                                             workers=workers,
                                             use_multiprocessing=use_multiprocessing,
                                             initial_epoch=self.init_epoch)
            elif self.model_type == 'AdvSeg':
                adv_cl = self.build_callbackList(use_tfboard=use_tfboard,
                                                 phase='AdversarialNet')
                seg_cl = self.build_callbackList(use_tfboard=use_tfboard,
                                                 monitor='val_seg_model_acc', 
                                                 phase='SegmentationNet')
                for i in range(num_round):
                    print('round {0} fitting seg_model'.format(i))
                    train_generator.phase = 'SegmentationNet'
                    valid_generator.phase = 'SegmentationNet'
                    self.adv_seg_model.fit_generator(generator=train_generator,
                                                     steps_per_epoch=seg_steps_per_epoch,
                                                     validation_data=valid_generator,
                                                     verbose=verbose,
                                                     epochs=(i+1)*seg_epochs,
                                                     callbacks=seg_cl,
                                                     workers=workers,
                                                     use_multiprocessing=
                                                     use_multiprocessing,
                                                     initial_epoch=i*seg_epochs)
                    print('round {0} fitting adv_model'.format(i))
                    train_generator.phase = 'AdversarialNet'
                    valid_generator.phase = 'AdversarialNet'
                    self.adv_model.fit_generator(generator=train_generator,
                                                 steps_per_epoch=adv_steps_per_epoch,
                                                 validation_data=valid_generator,
                                                 verbose=verbose,
                                                 epochs=(i+1)*adv_epochs,
                                                 callbacks=adv_cl,
                                                 workers=workers,
                                                 use_multiprocessing=use_multiprocessing,
                                                 initial_epoch=i*adv_epochs)
    
    def build_callbackList(self, use_tfboard=True, monitor=None, phase=None):
        if self.model_type == None:
            raise ValueError('model is not built yet, please build Segmentation or AdvSeg')
        else:
            path = './{0}/{1}'.format(self.model_type, self.dtype)

        # Model Checkpoints
        if monitor is None:
            callbackList = []
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            filepath=path+'/weights-{epoch:02d}-{'+'{0}'.format(monitor)+':.2f}.hdf5'
            checkpoint = ModelCheckpoint(filepath,
                                         monitor=monitor,
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='max')

            # Bring all the callbacks together into a python list
            callbackList = [checkpoint]
                    
        # Tensorboard
        if use_tfboard:
            if phase is None:
                tfpath = './logs/{0}/{1}'.format(self.model_type, self.dtype)
            else:
                tfpath = './logs/{0}/{1}/{2}'.format(self.model_type, phase, self.dtype)
            tensorboard = TrainValTensorBoard(log_dir=tfpath)
            callbackList.append(tensorboard)
        return callbackList
        
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