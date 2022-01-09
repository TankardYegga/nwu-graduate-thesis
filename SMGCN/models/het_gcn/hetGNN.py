# encoding=utf-8
from base_gnn import *
import tensorflow as tf
from  tensorflow import contrib

class HetGNN(BASE_GNN):
    def __init__(self, **kwargs):
        self.model_type=''
        self.embed = self.initial_embed_size
        super(HetGNN, self).__init__(**kwargs)
        self._train()

    def _init_weights(self):
        """ 需要三类参数，每类参数针对sympt和herb有所不同，总共3*2=6
        一是内容聚合的BiLSTM，二是同种类型聚合的BiLSTM，三是异质节点类型聚合的转换矩阵
        """
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        # 根据是否有pretrain的数据来初始化症状和草药嵌入
        if self.pretrain_data == None:
            all_weights['sympt_embeddings'] = tf.Variable(initializer([self.n_symptoms,self.initial_embed_size]),
                                                          name='sympt_embeddings')
            all_weights['herb_embeddings'] = tf.Variable(initializer([self.n_herbs, self.initial_embed_size]),
                                                         name='herb_embeddings')
            print("Initialize embeddings using xavier!")
        else:
            all_weights['sympt_embeddings'] = tf.Variable(initial_value=self.pretrain_data['sympt_embeddings'],
                                                          dtype=np.float32, trainable=True, name='sympt_embeddings')
            all_weights['herb_embeddings'] = tf.Variable(initial_value=self.pretrain_data['herb_embeddings'],
                                                         dtype=np.float32, trainable=True, name='herb_embeddings')
            print("Initialize embeddings using pretrain data")

        all_weights['sympt_content_agg_lstmcell_fw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)
        all_weights['sympt_content_agg_lstmcell_bw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)
        all_weights['sympt_neigh_agg_lstmcell_fw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)
        all_weights['sympt_neigh_agg_lstmcell_bw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)
        
        all_weights['herb_content_agg_lstmcell_fw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)
        all_weights['herb_content_agg_lstmcell_bw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)
        all_weights['herb_neigh_agg_lstmcell_fw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)
        all_weights['herb_neigh_agg_lstmcell_bw'] = tf.contrib.rnn.BasicLSTMCell(num_uints=self.embed/2)

        all_weights['herb_neigh_att'] = tf.Variable(tf.ones(self.embed*2, 1))
        all_weights['sympt_neigh_att'] = tf.Variable(tf.ones(self.embed*2, 1))
        
        return all_weights

    def _create_het_gnn_embed(self):
        pass

    def _create_placeholders(self):
        pass

    def _inference(self):
        pass
    
    def _train(self):
        pass

    """
    症状节点的属性嵌入有随机初始化的症状节点嵌入、能治疗该种症状的所有草药节点嵌入的平均值
    """
    def sympt_content_aggr(self):
        #  思路主要是将sympt节点的节点嵌入和对应草药嵌入进行维度拼接
        #  这里与hetGNN论文所不同的地方是原论文是有初始的paper、author、venue节点的初始嵌入的，而这里我们是没有的
        #  这里可以利用症状-草药的邻接矩阵来得到
        pass
    """
    草药节点的属性嵌入有随机初始化的草药节点嵌入、其能治疗的所有症状节点嵌入的平均值
    """
    def herb_content_aggr(self):
        # 主要是将herb节点的节点嵌入和对应的症状嵌入进行维度拼接
        pass

    def node_neigh_aggr(self):
        pass

    def node_het_aggr(self):
        pass

