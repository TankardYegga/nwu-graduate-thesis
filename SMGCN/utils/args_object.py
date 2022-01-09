class ArgsObject(object):
    def __init__(self, data_path='../data/', epoch=500, pretrain=0, gpu_id=0, initial_embed_size=64,
                 weight_size='[128, 128, 128]', sympt_threshold=10, herb_threshold=10, verbose=1,
                 Ks='[5,10,20,30,40]', node_dropout_flag=0, node_dropout='[0.0, 0.0, 0.0]',
                 mess_dropout='[0.0, 0.0, 0.0]', batch_size=1024, lr=0.001, regs='[1e-5,1e-5,1e-2]'
                 ):
        self.__data_path = data_path
        self.__epoch = epoch
        self.__pretrain = pretrain
        self.__gpu_id = gpu_id
        self.__initial_embed_size = initial_embed_size
        self.__weight_size = weight_size
        self.__sympt_threshold = sympt_threshold
        self.__herb_thresholdherb_threshold = herb_threshold
        self.__verbose = verbose
        self.__Ks = Ks
        self.__node_dropout_flag = node_dropout_flag
        self.__node_dropout = node_dropout
        self.__mess_dropout = mess_dropout
        self.__batch_size = batch_size
        self.__lr = lr
        self.__regs = regs

    @property
    def data_path(self):
        return self.__data_path
    
    @data_path.setter
    def data_path(self, value):
        self.__data_path = value

    @property
    def epoch(self):
        return self.__epoch

    @epoch.setter
    def epoch(self, value):
        self.__epoch = value

    @property
    def pretrain(self):
        return self.__pretrain

    @pretrain.setter
    def pretrain(self, value):
        self.__pretrain = value

    @property
    def gpu_id(self):
        return self.__gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        self.__gpu_id = value
    
    @property
    def initial_embed_size(self):
        return self.__initial_embed_size

    @initial_embed_size.setter
    def initial_embed_size(self, value):
        self.__initial_embed_size = value
    
    @property
    def weight_size(self):
        return self.__weight_size

    @weight_size.setter
    def weight_size(self, value):
        self.__weight_size = value
    
    @property
    def sympt_threshold(self):
        return self.sympt_threshold

    @sympt_threshold.setter
    def sympt_threshold(self, value):
        self.__sympt_threshold = value

    @property
    def herb_threshold(self):
        return self.__herb_threshold

    @herb_threshold.setter
    def herb_threshold(self, value):
        self.__herb_threshold = value

    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, value):
        self.__verbose = value
    
    @property
    def Ks(self):
        return self.__Ks

    @Ks.setter
    def Ks(self, value):
        self.__Ks = value

    @property
    def node_dropout_flag(self):
        return self.node_dropout_flag

    @node_dropout_flag.setter
    def node_dropout_flag(self, value):
        self.__node_dropout_flag = value
    
    @property
    def node_dropout(self):
        return self.__node_dropout

    @node_dropout.setter
    def node_dropout(self, value):
        self.__node_dropout = value

    @property
    def mess_dropout(self):
        return self.__mess_dropout

    @mess_dropout.setter
    def mess_dropout(self, value):
        self.__mess_dropout = value

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value
    
    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, value):
        self.__lr = value

    @property
    def regs(self):
        return self.__regs

    @regs.setter
    def regs(self, value):
        self.__regs = value







