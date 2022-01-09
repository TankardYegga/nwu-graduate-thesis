# pinsage获取所有item的嵌入表示，并在每个用户最近交互过的单个item组成的集合上进行推荐
# 利用的item之间的交互得分
# 但是在这里我们需要获取所有草药和所有症状的嵌入表示
# 并利用多标签损失函数来进行计算

# 我们这里有没有必要建立双向的二部图结构？
# 原先pinsage获取item嵌入是通过PinSage模型优化Margin Hinge Loss来实现的
# 每个迭代选取seeds为items，然后以这些items去进行二阶传播并保留传播过程中的二部图块结构
# 最后得到的就是seeds中所有tems的二阶传播或者说是二阶更新后的嵌入表示了
# 多次迭代能够所有的seeds尽可能地包括所有的items节点

# 这里需要重点解决的问题有
# 1. 模块的复用，因为症状和草药都通过同样的结构来获得嵌入
# 2. 到底应该是在以批量方式从所有症状节点或者草药节点获取嵌入表示后再进行训练
#    还是说 训练minibatch个症状节点或者草药节点后直接进行训练呢
# 3. 这些的训练参数有哪些，又在哪里具体说明呢
# 4.这里存在信息泄露问题吗？

# 关于原先PinSage模型要做出的相应改进
# 原模型包括线性映射器，多层卷积模块、计算损失模块
# 1.线性映射器   我们这里没有太多属性，就只有症状和草药节点的id而已，
#                所以类似于使用nn.embedding？还是使用xavier初始化
# 2.多层卷积模块 基本一致

from .graph_builder import GraphBuilder
from .sampler_module import *
from models.base_gnn import BASE_GNN
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

import dgl
import dgl.function as fn
import numpy as np
from .convolve_module import *


class PinSage(BASE_GNN):

    def __init__(self, data_generator, args):
        super().__init__(data_generator, args)
        self.model_type='pinsage'
        self._create_graph()
        self._create_convolves()
        self._train()


    def _create_graph(self):
        self.g_builder = GraphBuilder(self.data_generator)
        self.g = self.g_builder.build()
        print('self.g is,', self.g)


    def _create_convolves(self):
        self.multi_conv_net = MultiConvolveNet(self.args.hidden_dim, 
                            self.args.num_layers)


    def _init_weights(self):
        all_weights = dict()

        # initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.glorot_uniform_initializer()
        initializer = tf.initializers.GlorotUniform()

        # 根据是否有pretrain的数据来初始化症状和草药嵌入
        if self.pretrain_data == None:
            all_weights['sympt_embeddings'] = tf.Variable(initializer([self.n_symptoms, self.initial_embed_size]),
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

        return all_weights


    def _create_pinsage_embed(self):
        self.g.nodes['sympt'].data['id'] = self.weights['sympt_embeddings']
        self.g.nodes['herb'].data['id'] = self.weights['herb_embeddings']

        # herb和sympt的不同共现次数能不能作为权重呢
        # self.g.edges['cure'].data[''] = 
        # self.g.edges['cure_by'].data[''] = 
        updated_sympt_embeddings = self.update_embeddings(self.n_symptoms, 'sympt', 'herb',
                                                          self.args.random_walk_length,
                                                          self.args.random_walk_restart_prob,
                                                          self.args.num_random_walks, 
                                                          self.args.num_neighbors,
                                                          self.args.num_layers)
        updated_herb_embeddings = self.update_embeddings( self.n_herbs, 'herb', 'sympt',
                                                          self.args.random_walk_length,
                                                          self.args.random_walk_restart_prob,
                                                          self.args.num_random_walks, 
                                                          self.args.num_neighbors,
                                                          self.args.num_layers)
        return updated_sympt_embeddings, updated_herb_embeddings


    def pass_msg_on_convolves(self, blocks):
        h_all = blocks[0].srcdata['id']
        h_dst = blocks[-1].dstdata['id']
        return h_dst + self.multi_conv_net(blocks, h_all)


    def update_embeddings(self, entity_num, entity_type, other_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):

        num_batches = entity_num // self.args.batch_size + 1
        entity_batches = []
        for i in range(num_batches):
            start_idx = self.args.batch_size * i
            end_idx = self.args.batch_size * (i+1)
            if end_idx > entity_num:
                end_idx = entity_num
            entity_batches.append(tf.range(start_idx, end_idx))

        entity_batch_embeddings = []

        type_neighbor_sampler = NeighborSampler(self.g, entity_type, other_type,
                 random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers
            )
       
        for entity_batch in entity_batches:
            entity_seeds = entity_batch
            # 为这部分seeds构建二部图结构块
            blocks = type_neighbor_sampler.sample_blocks(entity_seeds)
            # 为这些block赋值相应的节点特征（此处是节点的嵌入表示）
            NeighborSampler.assign_features_to_blocks(blocks, self.g, entity_type, False)
            # 在这些block上进行信息的传播操作
            # 得到更新后的节点特征 
            updated_entity_embeddings = self.pass_msg_on_convolves(blocks)
            print(" updated_entity_embeddings is:",  updated_entity_embeddings)
            entity_batch_embeddings.append(updated_entity_embeddings)

        entity_batch_embeddings = tf.concat(entity_batch_embeddings, 0)
        return entity_batch_embeddings


    def _create_placeholders(self):
        """
          等待运行时动态填充的数据
        """
        self.sympt_set_sample_mat = tf1.placeholder(dtype=np.float32, shape=(None, self.n_symptoms))
        self.herb_set_sample_mat = tf1.placeholder(dtype=np.float32, shape=(None, self.n_herbs))
        self.sympt_set_normalized_mat = tf1.placeholder(dtype=np.float32, shape=(None, None))


    def _inference(self):
        # 获取每个样本中的症状集嵌入向量之和然后做平均操作
        self.sympt_set_embeddings_sum = tf.matmul(self.sympt_set_sample_mat, self.sympt_embeddings)
        self.mean_sympt_set_embedding = tf.matmul(self.sympt_set_normalized_mat, self.sympt_set_embeddings_sum)

        # 将获得的平均症状嵌入通过MLP获得一个综合的综合症嵌入表示
        self.syndrome_embedding = tf.nn.relu(tf.matmul(self.mean_sympt_set_embedding, self.weights['MLP_W']) + \
                                             self.weights['MLP_b'])

        # 将综合的综合症嵌入表示与草药嵌入矩阵做交互，得到大小为|H|的预测向量
        self.prediction_herb_mat = tf.matmul(self.syndrome_embedding, self.herb_embeddings, transpose_a=False, transpose_b=True)

        self.embed_loss, self.reg_loss = self.create_multilabel_loss(self.prediction_herb_mat,
                                                                     self.herb_set_sample_mat)

        self.total_loss = self.embed_loss + self.reg_loss
        

    def _train(self):
        self.sympt_embeddings, self.herb_embeddings = self._create_pinsage_embed()
        self._create_placeholders()
        self._inference()
        global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss,
                                                                          global_step=global_step)


    def create_multilabel_loss(self, prediction_herb_mat, herb_set_sample_mat):
        """
        1. 计算ground_truth草药集
        """
        ground_truth_herb_mat = herb_set_sample_mat

        """
        2.计算损失的权重向量
        """
        # 计算每种草药在处方中出现的频次
        herb_freq_vector = list(np.zeros(dtype=np.float32, shape=(self.n_herbs,)))
        for prescrp in self.data_generator.train_prescrp_list:
            herb_set = prescrp.herbs
            for herb in herb_set:
                herb_freq_vector[int(herb)] += 1.
        # 根据频次计算权重
        herb_freq_vector = np.asarray(herb_freq_vector)
        herb_weight_vector = (max(herb_freq_vector) + 1.0) /( herb_freq_vector + 1.0)
        herb_weight_vector = herb_weight_vector.reshape([1,self.n_herbs]).astype(dtype=np.float32)
        print('herb_weight_vector', herb_weight_vector)
        print('herb_weight_vector type:', type(herb_freq_vector))
        print('herb_weight_vector shape:', herb_freq_vector.shape)

        """
        3.计算损失
        """
        embed_loss = tf.matmul(tf.square(tf.subtract(prediction_herb_mat, ground_truth_herb_mat)), herb_weight_vector,
                               transpose_a=False, transpose_b=True)
        embed_loss = tf.reduce_mean(embed_loss)
    
        # 参数的正则化损失
        regularizer = tf.keras.regularizers.l2(0.5)
        reg_loss = 0.0
        for param in self.weights.values():
            reg_loss += regularizer(param)

        # batch_len = prediction_herb_mat.get_shape().as_list()
        # print('type is', type(prediction_herb_mat))
        # print(batch_len)
        return embed_loss, reg_loss / self.batch_size



