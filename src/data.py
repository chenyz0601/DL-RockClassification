import numpy as np
from osgeo import gdal
import glob
import keras
# from sklearn.model_selection import train_test_split

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_IDs, Y_IDs, batch_size=32, shuffle=True, dtype='all'):
        'Initialization'
        self.dtype = dtype
        self.X_IDs = X_IDs
        self.Y_IDs = Y_IDs
        if len(self.X_IDs) != len(self.Y_IDs):
            raise ValueError('imgs and labels are not matched')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_IDs) / self.batch_size))

    def getitem(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X_IDs_temp = [self.X_IDs[k] for k in indexes]
        Y_IDs_temp = [self.Y_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(X_IDs_temp, Y_IDs_temp)

        return X, y
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X_IDs_temp = [self.X_IDs[k] for k in indexes]
        Y_IDs_temp = [self.Y_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(X_IDs_temp, Y_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, X_IDs_temp, Y_IDs_temp):
        'Generates data containing batch_size samples' 
        # X_out : (n_samples, *dim, n_channels)
        # Y_out : (n_samples, *dim, n_classes)
        # Initialization
        X_out = []
        Y_out = []
        if self.dtype == 'all':
            for i in range(len(X_IDs_temp)):
                X_out.append(np.transpose(np.load(X_IDs_temp[i]), [1,2,0]))
                Y_out.append(np.transpose(np.load(Y_IDs_temp[i]), [1,2,0]))
        elif self.dtype == 'sent':
            for i in range(len(X_IDs_temp)):
                X_out.append(np.transpose(np.load(X_IDs_temp[i])[:10,:,:], [1,2,0]))
                Y_out.append(np.transpose(np.load(Y_IDs_temp[i]), [1,2,0]))
        else:
            raise ValueError('unknown dtype, should be all or sent')
        return np.asarray(X_out), np.asarray(Y_out)

class Data:
    
    def __init__(self, path, random=False):
        """
        input:
            path: path to the block
            max_num: int, num of samples
            random: bool, to load samples randomly or from 0 to num_max
        """
        self.X = sorted(glob.glob(path+"images/images/*.tif"))
        self.Y = sorted(glob.glob(path+"labels/images/*.tif"))
        if len(self.X) != len(self.Y):
            raise ValueError('imgs and labels are not matched')
      
    def get_MinMax(self, path):
        file = gdal.Open(path)
        min_list = []
        max_list = []
        for i in range(1, file.RasterCount+1):
            if i < 11:
                min_list.append(0.)
                max_list.append(10000.)
            else:
                min_list.append(file.GetRasterBand(i).GetMinimum())
                max_list.append(file.GetRasterBand(i).GetMaximum())
        return min_list, max_list
        
    def img_to_array(self, input_file, min_list=None, max_list=None, dtype='float32'):
        """
        convert a raster tile into numpy array
        input:
            input_file: string, path a raster(.tif)
            normalizer: double, if input is labels with 0 or 1, it's 1.
                                if input is sentinal data (reflectance), then it's 10000.
            dtype: string, data type, default as 'float32'
        return:
            arr: numpy array, shape is [dim_y, dim_x, num_bands]
        """
        file = gdal.Open(input_file)
        if min_list != None:
            bands = [(np.array(file.GetRasterBand(i).ReadAsArray()).astype(dtype) - min_list[i-1]) / (max_list[i-1] - min_list[i-1]) for i in range(1, file.RasterCount + 1)]
        else:
            bands = [np.array(file.GetRasterBand(i).ReadAsArray()).astype(dtype) for i in range(1, file.RasterCount + 1)]
        arr = np.stack(bands, axis=2)
        arr[arr>1.] = 0.
        return arr

    def get_XY(self, min_list, max_list, start=0, num=10, as_arr=False, random=False):
        """
        function: load max_num of XY into lists
        output: list of numpy arrays, X (images) and Y (labels)
        """
        X_out = []
        Y_out = []
        
        if random:
            idx = np.random.choice(list(range(len(self.X))), num, replace=False)
            print('randomly loading {0} tiles from {1} tiles'.format(num, len(self.X))) 
        else:
            idx = list(range(start, start+num))
            print('loading {0} - {1} image tiles'.format(start, start+num-1))

        for i in idx:
            X_out.append(self.img_to_array(self.X[i], min_list=min_list, max_list=max_list))
            Y_out.append(self.img_to_array(self.Y[i], dtype='int'))
        
        X_remove = [self.X[i] for i in idx]
        Y_remove = [self.Y[i] for i in idx]
        
        for i in range(len(X_remove)):
            self.X.remove(X_remove[i])
            self.Y.remove(Y_remove[i])
        
        if as_arr:
            return np.asarray(X_out), np.asarray(Y_out)
        else:
            return X_out, Y_out
        
    """
    def trn_tst_split(self, test_rate, random_seed):
        X, Y = self.get_XY()
        X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, train_size=1-test_rate, test_size=test_rate, random_state=random_seed)
        return np.asarray(X_trn), np.asarray(X_tst), np.asarray(Y_trn), np.asarray(Y_tst)
    """