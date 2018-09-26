from src.DeepRock.models import AdvSeg
# from src.DeepRock.data import DataGenerator
from src.DeepRock.utils import split_trn_vld_tst, get_XY
import matplotlib.pyplot as plt
import numpy as np
import sys

def train(X_trn, Y_trn, n_ch_list=[64, 64, 64, 64], lr=1e-3,
          epochs=100, batch_size=8, dtype='sent'):
    conv = AdvSeg(dtype=dtype)
    conv.build_SegmentationNet(lr=lr, n_ch_list=n_ch_list)
    x, y = get_XY(X_trn, Y_trn)
    conv.fit_model(x, y, num_epochs=epochs)

def main(args):
    X_trn, Y_trn, _, _, _, _ = split_trn_vld_tst('./data/train/',
            vld_rate=0, tst_rate=0.1, seed=10)
    # fine tune the learning rate
    train(X_trn, Y_trn, n_ch_list=[64, 64, 64, 64], lr=float(args[2]),
            batch_size=8, epochs=int(args[1]), dtype=args[0])

if __name__ == '__main__':
    main(sys.argv[1:])
