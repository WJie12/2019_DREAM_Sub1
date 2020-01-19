class Config():
    def __init__(self):
        debug=False
        self.mode='train'
        self.ori_dir='./data/test/train/' if debug else './train/'
        self.dir='./data/test/split/' if debug else './data/complete/split/'
        self.train_dir=self.dir
        self.test_dir=self.dir+'test/'
        self.ori_train=self.ori_dir
        self.model_dir=self.dir+'model/dx'
        self.log_dir=self.dir+'log/'
        self.condition='EGF'
        self.split_dir='./data/test/split/' if debug else './data/complete/split/'
        self.use_label=True
        self.pred_label=False
        self.train_num=1 if debug else 21
        self.batch_size=128
        self.gene_num=37
        self.label_num=6
        self.hidden_size=20 if debug else 128
        self.label_emb_size=32
        self.lr=1e-9
        self.valid_list=[12,7,30,16,28]
        # self.valid_list = ['p.ERK', 'p.Akt.Ser473.', 'p.S6', 'p.HER2', 'p.PLCg2']
        self.epoch=2 if debug else 15
        self.max_length=10
        self.max_time=100
        self.use_position_embedding=False
        self.model_type='static'
        self.pre_train_epoch=1 if debug else int(self.epoch*0.3)
        self.train_split_param=10 if debug else 0.0005

