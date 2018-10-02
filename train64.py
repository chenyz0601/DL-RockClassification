from src.DeepRock.models import AdvSeg
from src.DeepRock.data import DataGenerator
from src.DeepRock.utils import split_trn_vld_tst, get_XY
import matplotlib.pyplot as plt
import numpy as np
import sys

def train(trn, vld, n_ch_list=[64, 64, 64, 64], lr=1e-3, epochs=100, dtype='sent'):
    conv = AdvSeg(dtype=dtype, dim_width=64, dim_height=64, num_labels=8)
    conv.build_SegmentationNet(lr=lr, n_ch_list=n_ch_list)
    conv.fit_model_generator(trn, vld, seg_epochs=epochs)

def main(args):
    # prepare data from 4 blocks
    X_trn, Y_trn, X_vld, Y_vld, _, _ = split_trn_vld_tst('./data/block1_64/', vld_rate=0.2, tst_rate=0.1, seed=10)
    
    trn = DataGenerator(X_trn, Y_trn, dtype=args[0], batch_size=32)
    vld = DataGenerator(X_vld, Y_vld, dtype=args[0], batch_size=32)
    # fine tune the learning rate
    train(trn, vld, n_ch_list=[64, 64], lr=5e-4, epochs=100, dtype=args[0])

if __name__ == '__main__':
    main(sys.argv[1:])
