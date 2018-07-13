import glob
import re
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Lambda, AlphaDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
# from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
import os, random

class myModel:
    
    def __init__(self, num_bands=10,
                 dim_width=256,
                 dim_height=256,
                 num_labels=10):
        self.num_bands = num_bands
        self.dim_width = dim_width
        self.dim_height = dim_height
        self.num_labels = num_labels
        self.Segmentor = None
        self.Segmentor_type = None
        self.model = None
        self.callbackList = None
        self.model_type = None
        self.init_epoch = 0
    
    def build_SegmentorNet(self, k_size = (3, 3),
                           n_ch_list=[64, 64, 64, 64],
                           k_init='lecun_normal',
                           activation='selu'):
        inp, outp = self.get_SegmentorNet(n_ch_list, k_size, k_init, activation)
        self.model = Model(inputs=[inp], outputs=[outp])
        self.model_type = 'Segmentor'
    
    def build_AdvSegNet(self, k_size = (3, 3),
                        n_ch_list=[64, 64, 64, 64],
                        k_init='lecun_normal',
                        activation='selu'):
        img_inp, pred_inp = self.get_SegmentorNet(n_ch_list, k_size, k_init, activation)
        label_shape = (self.dim_width, self.dim_height, self.num_labels)
        label_inp = Input(label_shape)
        out_true = self.get_AdversarialNet(img_inp, label_inp)
        out_false = self.get_AdversarialNet(img_inp, pred_inp)
        self.model = Model(inputs=[img_inp, label_inp], outputs=[self.Segmentor.outputs, out_true, out_false])
        self.model_type = 'AdvSeg'
        
    def compile_model(self, scale=1e-2, lr=1e-3, verbose=True):
        adam = keras.optimizers.adam(lr=lr)
        if self.model_type == 'Segmentor':
            print('compiling Segmentor only ...')
            loss = 'categorical_crossentropy'
            # build the whole computational graph with model, loss and optimizer
            # 'accuracy' is defaultly categorical_accuracy
            self.model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
            # print parameters of each layer
            if verbose:
                print(self.model.summary())
        elif self.model_type == 'AdvSeg':
            print('compiling Segmentor with Adversarial net ...')
            loss = self.get_AdvSegLoss(scale)
            self.model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
            # print parameters of each layer
            if verbose:
                print(self.model.summary())
        else:
            raise ValueError('no model to be compiled!')
                    
    def fit_Segmentor_generator(self, train_generator,
                                valid_generator,
                                verbose=1,
                                workers=1,
                                use_multiprocessing=False,
                                use_tfboard=True,
                                num_epochs=10):
        if self.model_type == 'Segmentor':
            self.build_callbackList(use_tfboard)
            self.model.fit_generator(generator=train_generator,
                                     validation_data=valid_generator,
                                     verbose=verbose,
                                     epochs=self.init_epoch+num_epochs,
                                     callbacks=self.callbackList,
                                     workers=workers,
                                     use_multiprocessing=use_multiprocessing,
                                     initial_epoch=self.init_epoch)
        else:
            raise ValueError('model type should be Segmentor!')
        
    def get_SegmentorNet(self, n_ch_list, k_size, k_init, activation):
        """
        input:
            num_bands, int, number of input channels
            n_ch_list, list, int numbers for num of channels of each layer
            num_labels, int, num of classes that each pixel will be assigned a prob, should be filters of last 'softmax' layer
            k_init, string, define the way of initializing the weights
            activation, string, define the activation function of intermediate layers
        output:
            keras model
        """

        if K.image_data_format() == 'channels_first':
            ch_axis = 1
            print('there might be a problem with softmax')
            input_shape = (self.num_bands, self.dim_width, self.dim_height)
        elif K.image_data_format() == 'channels_last':
            ch_axis = 3
            input_shape = (self.dim_width, self.dim_height, self.num_bands)


        inp = Input(input_shape)
        encoder = inp
        list_encoders = []

    # summary image requires num of channels to be 1, 3 or 4
    #     if use_tfboard:
    #         tf.summary.image(name='input', tensor=inp)  

        print('building Unet ...')
        print(n_ch_list)
        # encoders
        for l_idx, n_ch in enumerate(n_ch_list):
            with K.name_scope('Encoder_block_{0}'.format(l_idx)):
                encoder = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder)
                encoder = AlphaDropout(0.1*l_idx, )(encoder)
                encoder = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 dilation_rate=(2, 2),
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder)
                list_encoders.append(encoder)
                # add maxpooling layer except the last layer
                if l_idx < len(n_ch_list) - 1:
                    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
                # if use_tfboard:
                    # tf.summary.histogram('conv_encoder', encoder)
        # decoders
        decoder = encoder
        dec_n_ch_list = n_ch_list[::-1][1:]
        print(dec_n_ch_list)
        for l_idx, n_ch in enumerate(dec_n_ch_list):
            with K.name_scope('Decoder_block_{0}'.format(l_idx)):
                l_idx_rev = len(n_ch_list) - 1 - l_idx
                decoder = concatenate([decoder, list_encoders[l_idx_rev]], axis=ch_axis)
                decoder = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 dilation_rate=(2, 2),
                                 kernel_initializer=k_init)(decoder)
                decoder = AlphaDropout(0.1*l_idx, )(decoder)
                decoder = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(decoder)
                decoder = Conv2DTranspose(filters=n_ch,
                                          kernel_size=k_size,
                                          strides=(2, 2), 
                                          activation=activation,
                                          padding='same',
                                          kernel_initializer=k_init)(decoder)

        # output layer should be softmax
        # instead of using Conv2DTranspose, Dense layer could also be tried
        outp = Conv2DTranspose(filters=self.num_labels,
                               kernel_size=k_size,
                               activation='softmax',
                               padding='same',
                               kernel_initializer='glorot_normal')(decoder)

        # summary image requires num of channels to be 1, 3 or 4
    #         if use_tfboard:
    #             tf.summary.image(name='output', tensor=outp)
        return inp, outp
    
    def get_AdversarialNet(self, inpX, inpY):
        ### TO DO ###
        pass
        return
    
    def get_AdvSegLoss(scale):
        return 
    
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

    ### TO DO: fit_generator ###
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