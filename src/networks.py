from keras.layers import Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import Input, AveragePooling2D, MaxPooling2D, Dropout, Lambda, AlphaDropout
from keras.layers.merge import concatenate
from keras import backend as K
from keras.models import Model

def SegmentationNet(inp, num_labels, n_ch_list, k_size, k_init, activation):
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
        concat_axis = 1
        print('there might be a problem with softmax, please set to channels_last')
    elif K.image_data_format() == 'channels_last':
        concat_axis = 3
    encoder = inp
    list_encoders = []

# summary image requires num of channels to be 1, 3 or 4
#     if use_tfboard:
#         tf.summary.image(name='input', tensor=inp)  

    print('building Segmentation U-net ...')
    print(n_ch_list)
    # encoders
    with K.name_scope('SegmentationNet'):
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
                decoder = concatenate([decoder, list_encoders[l_idx_rev]], axis=concat_axis)
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
        outp = Conv2DTranspose(filters=num_labels,
                               kernel_size=k_size,
                               activation='softmax',
                               padding='same',
                               kernel_initializer='glorot_normal')(decoder)

    # summary image requires num of channels to be 1, 3 or 4
#         if use_tfboard:
#             tf.summary.image(name='output', tensor=outp)
    return Model(inputs=[inp], outputs=[outp])
    
def AdversarialNet(inpX, inpY, ch_list,
                   k_size, k_init, activation, br_ch):
    print('building Adversarial convolutional net ...')
    if K.image_data_format() == 'channels_first':
        concat_axis = 1
        print('there might be a problem with softmax, please set to channels_last')
    elif K.image_data_format() == 'channels_last':
        concat_axis = 3
    with K.name_scope('AdversarialNet'):
        with K.name_scope('img_input_conv'):
            X = Conv2D(filters=br_ch,
                       kernel_size=k_size,
                       activation=activation,
                       padding='same',
                       kernel_initializer=k_init)(inpX)
        with K.name_scope('label_input_conv'):
            Y = Conv2D(filters=br_ch,
                       kernel_size=k_size,
                       activation=activation,
                       padding='same',
                       kernel_initializer=k_init)(inpY)
        encoder = concatenate([X, Y], axis=concat_axis)
        for l_idx, n_ch in enumerate(ch_list):
            with K.name_scope('AdvNet_block_{0}'.format(l_idx)):
                encoder = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder)
                # encoder = AlphaDropout(0.1*l_idx, )(encoder)
                # add maxpooling layer except the last layer
                if l_idx < len(ch_list) - 1:
                    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
        encoder = Flatten()(encoder)
        outp = Dense(1, activation='sigmoid')(encoder)
    return Model(inputs=[inpX, inpY], outputs=outp)