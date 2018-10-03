from src.DeepRock.models import AdvSeg
from src.DeepRock.data import DataGenerator
from src.DeepRock.utils import split_trn_vld_tst, get_XY
import numpy as np
import sys

def train(trn, vld, n_ch_list=[64, 64, 64, 64], lr=1e-3, epochs=100, dtype='sent'):
    conv = AdvSeg(dtype=dtype)
    conv.build_SegmentationNet(lr=lr, n_ch_list=n_ch_list)
    conv.fit_model_generator(trn, vld, seg_epochs=epochs)

def main(args):
    # prepare data from 4 blocks
    X_trn1, Y_trn1, X_vld1, Y_vld1, _, _ = split_trn_vld_tst('./data/block1/train/', vld_rate=0.2, tst_rate=0, seed=10)
    X_trn2, Y_trn2, X_vld2, Y_vld2, _, _ = split_trn_vld_tst('./data/block2/train/', vld_rate=0.2, tst_rate=0, seed=10)
    X_trn3, Y_trn3, X_vld3, Y_vld3, _, _ = split_trn_vld_tst('./data/block3/train/', vld_rate=0.2, tst_rate=0, seed=10)
    X_trn4, Y_trn4, X_vld4, Y_vld4, _, _ = split_trn_vld_tst('./data/block4/train/', vld_rate=0.2, tst_rate=0, seed=10)
    X_trn = X_trn1 + X_trn2 + X_trn3 + X_trn4
    Y_trn = Y_trn1 + Y_trn2 + Y_trn3 + Y_trn4
    X_vld = X_vld1 + X_vld2 + X_vld3 + X_vld4
    Y_vld = Y_vld1 + Y_vld2 + Y_vld3 + Y_vld4
    
    trn = DataGenerator(X_trn, Y_trn, dtype=args[0], batch_size=8)
    vld = DataGenerator(X_vld, Y_vld, dtype=args[0], batch_size=8)
    # fine tune the learning rate
    train(trn, vld, n_ch_list=[64, 64, 64, 64], lr=float(args[2]), epochs=int(args[1]), dtype=args[0])

if __name__ == '__main__':
    main(sys.argv[1:])
