import tensorflow as tf
import numpy as np

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


class BASE_GNN(object):
    def __init__(self, data_generator, args):
        self.data_generator = data_generator
        self.args = args
        self._load_data()
        self.weights = self._init_weights()
        
        
    def _load_data(self):
        if self.args.pretrain == -1:
            pretrain_data = load_pretrained_data()
        else:
            pretrain_data = None
        self.pretrain_data = pretrain_data

        self.n_symptoms = self.data_generator.n_symptoms
        self.n_herbs = self.data_generator.n_herbs

        self.initial_embed_size = self.args.initial_embed_size
        self.batch_size = self.args.batch_size

        self.lr = self.args.lr
        self.weight_size = eval(self.args.weight_size)
        self.hidden_size = eval(self.args.hidden_size)
        self.n_layers = len(self.weight_size)

        self.node_dropout_flag = self.args.node_dropout_flag
        self.node_dropout = eval(self.args.node_dropout)
        self.mess_dropout = eval(self.args.mess_dropout)
        self.regs = eval(self.args.regs)
        self.decay = self.regs[0]
        self.verbose = self.args.verbose

    def _init_weights(self):
        raise NotImplementedError()

    def _train(self):
        raise  NotImplementedError()





