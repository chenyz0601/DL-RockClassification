import numpy as np
from osgeo import gdal
import glob
from sklearn.model_selection import train_test_split

class Data:
    
    def __init__(self, path, start=0, num=10, random=False):
        """
        input:
            path: path to the block
            max_num: int, num of samples
            random: bool, to load samples randomly or from 0 to num_max
        """
        self.path_to_block = path
        self.N = num + start
        self.start = start
        self.random = random
      
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
        return arr

    def load_XY(self, min_list, max_list, as_arr=False):
        """
        function: load max_num of XY into lists
        output: list of numpy arrays, X (images) and Y (labels)
        """
        X_out = []
        Y_out = []
        X_path = sorted(glob.glob(self.path_to_block+"images/images/*.tif"))
        Y_path = sorted(glob.glob(self.path_to_block+"labels/images/*.tif"))
        if len(X_path) != len(Y_path):
            raise ValueError('imgs and labels are not matched')
        if self.random:
            idx = np.random.randint(0, len(X_path), self.N)
        else:
            if self.N == None:
                self.N = len(X_path)
            idx = range(self.start, self.N)

        for i in idx:
            X_out.append(self.img_to_array(X_path[i], min_list=min_list, max_list=max_list))
            Y_out.append(self.img_to_array(Y_path[i], dtype='int'))
        if as_arr:
            return np.asarray(X_out), np.asarray(Y_out)
        else:
            return X_out, Y_out
        
    def trn_tst_split(self, test_rate, random_seed):
        """
        input:  test_rate, double, between 0 and 1,
                random_seed, randomness to generate tst and trn data
        output: lists of train and test datasets
        """
        X, Y = self.load_XY()
        X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, train_size=1-test_rate, test_size=test_rate, random_state=random_seed)
        return np.asarray(X_trn), np.asarray(X_tst), np.asarray(Y_trn), np.asarray(Y_tst)