# -*- coding:utf-8 -*-


class hyperparameter():
    def __init__(self):
        self.Learning_rate = 0.0001
        self.Epoch = 10
        self.Batch_size = 16
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.001
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [5,8,10]
        self.drug_kernel = [3, 5, 6]
        self.conv = 40
        self.char_dim = 64
        self.loss_epsilon = 1