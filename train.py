from src.DeepRock.models import AdvSeg
from src.DeepRock.data import DataGenerator
from src.DeepRock.utils import split_trn_vld_tst
import matplotlib.pyplot as plt
import numpy as np

def train(X_trn, Y_trn, X_vld, Y_vld, n_ch_list=[64, 64, 64, 64], lr=1e-3, epochs=100, batch_size=8, dtype='sent'):
    conv = AdvSeg(dtype=dtype)
    conv.build_SegmentationNet(lr=lr)
    trn_data = DataGenerator(X_trn, Y_trn, batch_size=batch_size, dtype=dtype, intype=conv.model_type, pred_fn=conv.predict)
    vld_data = DataGenerator(X_vld, Y_vld, batch_size=batch_size, dtype=dtype, intype=conv.model_type, pred_fn=conv.predict)
    conv.fit_model_generator(trn_data, vld_data, seg_epochs=epochs, seg_steps_per_epoch=None, save_weights=True)

if __name__ == '__main__':
    X_trn, Y_trn, X_vld, Y_vld, _, _ = split_trn_vld_tst('./data/train/', vld_rate=0.2, tst_rate=0, seed=10)
    # fine tune the learning rate
    train(X_trn, Y_trn, X_vld, Y_vld, n_ch_list=[64, 64, 64, 64], lr=0.0005, batch_size=8)