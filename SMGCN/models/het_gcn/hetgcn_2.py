from utils.mat_operation import normalized_adj_single
from .base_gnn import *
import tensorflow as tf
import numpy as np

# 与第一版的区别在于卷积层参数中的隐藏层维度和输出层维度
class HetGCN_2(BASE_GNN):
    def __init__(self, **kwargs):
        self.model_type='hetgcn'
        self.sess = tf.InteractiveSession()
        super(HetGCN_2, self).__init__(**kwargs)
        self._train()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data == None:
            all_weights['sympt_embeddings'] = tf.Variable(initializer([self.n_symptoms, self.initial_embed_size]),
                                                        name='sympt_embeddings'
                                                    )
            all_weights['herb_embeddings'] = tf.Variable(initializer([self.n_herbs, self.initial_embed_size]),
                                                    name='herb_embeddings'
            )
            print('Initializing embeddings using xavier ...')
        else:
            all_weights['sympt_embeddings'] = tf.Variable(initial_value=self.pretrain_data['sympt_embeddings'], dtype=np.float32,
                                                    trainable=True, name='sympt_embeddings')
            all_weights['herb_embeddings'] = tf.Variable(initial_value=self.pretrain_data['herb_embeddings'], dtype=np.float32,
                                                    trainable=True, name='herb_embeddings') 
            print('Initializing embeddings using pretrained data ...')

        self.weight_size_list = [self.initial_embed_size] + self.weight_size


        # 每一个HetGCN层都需要四个参数
        # 1 每层用于构造邻居消息的T
        # 2 每层用于节点嵌入和邻居嵌入合并后的转换权重矩阵
        # 3、4 每层不同类型邻居进行聚合所需的注意力参数Watt 和 Z
        for k in range(self.n_layers):
            all_weights['neig_msg_construct_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k]]),
                name = 'neig_msg_construct_%d' % k,
            )
            all_weights['weight_attention_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k] * 2, self.weight_size_list[k]]),
                name = 'weight_attention_%d' % k,
            )
            all_weights['z_attention_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k],  1]),
                name = 'z_attention_%d' % k,
            )
            all_weights['neigh_combine_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k] * 2, self.weight_size_list[k+1]]),
                name = 'neigh_combine_%d' % k,
            )

        return all_weights

    # 在hetgcn上进行信息传播，获取更新后的草药和症状节点的嵌入
    def _create_hetgcn_embed(self):
        # 获取初始的草药和症状-嵌入矩阵
        ego_embeddings = tf.concat([self.weights['sympt_embeddings'], 
                                    self.weights['herb_embeddings']], axis=0)

        """
        获取并构建（度矩阵的倒数）归一化的同质邻接矩阵和异质邻接矩阵
        """
        # 获取草药-草药共现矩阵和症状-症状共现矩阵
        sympt_sympt_cocur_mat = self.data_generator.get_adj_mat()[1]
        herb_herb_cocur_mat = self.data_generator.get_adj_mat()[2]
        # 获取草药-草药邻接矩阵和症状-症状邻接矩阵
        sympt_sympt_adj_mat = self.get_adj_mat_by_threshold(sympt_sympt_cocur_mat, self.args.sympt_threshold)
        herb_herb_adj_mat = self.get_adj_mat_by_threshold(herb_herb_cocur_mat, self.args.herb_threshold)

        self.hete_adj_R = self.data_generator.A
        self.homo_adj_R = np.zeros(shape=(self.n_symptoms + self.n_herbs, self.n_symptoms + self.n_herbs), 
                            dtype=np.float32)
        self.homo_adj_R[:self.n_symptoms,:self.n_symptoms] = sympt_sympt_adj_mat
        self.homo_adj_R[-self.n_herbs:, -self.n_herbs:] = herb_herb_adj_mat

        normalized_hete_adj_R  = tf.convert_to_tensor(normalized_adj_single(self.hete_adj_R, method=1))
        normalized_homo_adj_R  = tf.convert_to_tensor(normalized_adj_single(self.homo_adj_R, method=1))
        
        """
        在每一层GCN上进行计算
        """
        for k in range(self.n_layers):
            # 构造要传播的消息
            constructed_msg = tf.matmul(ego_embeddings, self.weights['neig_msg_construct_%d' % k])
            print('constructed msg', self.sess.run(tf.shape(constructed_msg)))

            # 构造同质邻居消息
            homo_neigh_msg = tf.matmul(normalized_homo_adj_R, constructed_msg)
            # 构造异质邻居消息
            hete_neigh_msg = tf.matmul(normalized_hete_adj_R, constructed_msg)
            # 拼接邻居消息
            neigh_msg_aggr = tf.stack([homo_neigh_msg, hete_neigh_msg], axis=1)

            # 拼接原始嵌入矩阵和邻居嵌入矩阵
            homo_aggr_msg = tf.concat([ego_embeddings, homo_neigh_msg], axis=1)
            hete_aggr_msg = tf.concat([ego_embeddings, hete_neigh_msg], axis=1)
            # 维度应该是[n_herbs + n_sympts, 2, （原始嵌入 + 邻居嵌入）的长度]
            aggr_msg = tf.stack([homo_aggr_msg, hete_aggr_msg], axis=1)
          
            print('aggr_msg shape:', self.sess.run(tf.shape(aggr_msg)) )

            watt = tf.tile(self.weights['weight_attention_%d' % k], [tf.shape(aggr_msg)[0], 1])
            watt = tf.reshape(watt, [tf.shape(aggr_msg)[0], tf.shape(self.weights['weight_attention_%d' % k])[0],
                    tf.shape(self.weights['weight_attention_%d' % k])[1]
               ])

            z_att = self.weights['z_attention_%d' % k][None, :, :]
            z_att = tf.tile(z_att, [tf.shape(aggr_msg)[0], 1, 1])

            # 应用Watt + z
            attention_weight = tf.matmul( tf.matmul(aggr_msg, watt), z_att)
            attention_weight = tf.nn.softmax( tf.transpose(attention_weight, (0,2,1)), axis=-1)

            # 回去加权后的邻居消息
            weighted_neigh_msg = tf.nn.tanh(tf.matmul(attention_weight, neigh_msg_aggr))
            weighted_neigh_msg = tf.reshape(weighted_neigh_msg, 
                    shape=(tf.shape(weighted_neigh_msg)[0], tf.shape(weighted_neigh_msg)[-1]))
            # weighted_neigh_msg = tf.cast(weighted_neigh_msg, tf.float32)

            # 将邻居消息和自身消息进行连接
            print('dtype of ego_embeddings:',  ego_embeddings.dtype, 'shape:', self.sess.run(tf.shape(ego_embeddings)))
            print('dtype of weighted_neight_msg:', weighted_neigh_msg.dtype, 'shape:', self.sess.run(tf.shape(weighted_neigh_msg)))
            concated_msg = tf.concat([ego_embeddings, weighted_neigh_msg],axis=1)
            print('concated_msg shape:', self.sess.run(tf.shape(concated_msg)))

            # 更新节点自身的嵌入
            ego_embeddings = tf.nn.tanh(tf.matmul(concated_msg, self.weights['neigh_combine_%d' % k]))
    
        # 返回在GCN上更新后的症状草药-嵌入矩阵
        sympt_embeddings, herb_embeddings = tf.split(ego_embeddings, [self.n_symptoms, self.n_herbs],
                                                    axis=0)
        return sympt_embeddings, herb_embeddings


    def _create_placeholders(self):
        self.sympt_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_symptoms))
        self.herb_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_herbs))
        self.sympt_set_normalized_mat = tf.placeholder(dtype=np.float32, shape=(None, None))

    def _inference(self):
        # 获取每个样本中的症状集嵌入向量之和然后做平均操作
        self.sympt_set_embeddings_sum = tf.matmul(self.sympt_set_sample_mat, self.sympt_embeddings)
        self.mean_sympt_set_embedding = tf.matmul(self.sympt_set_normalized_mat, self.sympt_set_embeddings_sum)
        self.syndrome_embedding = self.mean_sympt_set_embedding

        # 将综合的综合症嵌入表示与草药嵌入矩阵做交互，得到大小为|H|的预测向量
        self.prediction_herb_mat = tf.matmul(self.syndrome_embedding, self.herb_embeddings, transpose_a=False, transpose_b=True)

        self.embed_loss, self.reg_loss = self.create_multilabel_loss(self.prediction_herb_mat,
                                                                     self.herb_set_sample_mat)

        self.total_loss = self.embed_loss + self.reg_loss

    def _train(self):
        self.sympt_embeddings, self.herb_embeddings = self._create_hetgcn_embed()
        self._create_placeholders()
        self._inference()
        global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss, global_step)

    def get_adj_mat_by_threshold(self, cocur_mat, threshold):
        adj_mat = np.zeros(dtype=np.float32, shape=cocur_mat.shape)
        for i in range(cocur_mat.shape[0]):
            for j in range(i+1, cocur_mat.shape[1]):
                if cocur_mat[i][j] >= threshold:
                    adj_mat[i][j] = 1.0
                    adj_mat[j][i] = 1.0
        return adj_mat

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
        regularizer = tf.contrib.layers.l2_regularizer(0.5)
        reg_loss = 0.0
        for param in self.weights.values():
            reg_loss += regularizer(param)

        # batch_len = prediction_herb_mat.get_shape().as_list()
        # print('type is', type(prediction_herb_mat))
        # print(batch_len)
        return embed_loss, reg_loss / self.batch_size