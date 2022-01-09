import sys
# sys.path.append('..')

from models.base_gnn import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
相比于SMGCN，没有MLP只有平均池化层 
"""
class SGE(BASE_GNN):
    def __init__(self, **kwargs):
        self.model_type='sge'
        super(SGE, self).__init__(**kwargs)
        self._train()

    def _init_weights(self):
        """
        # 初始化的症状嵌入和草药嵌入维度 2
        # 每层分别针对症状和草药的消息构造和消息聚合矩阵 4 * n_layers
        # 二部图GCN的层数 n_layers
        # 二部图GCN每层的输出维度
        # 协同图卷积的权重参数 Vs和Vh
        # 一层MLP的权重参数矩阵 'MLP_W' 'MLP_b'
        :return:
        """
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.initializers.GlorotUniform()

          # 维度参数列表：输入维度 + 各个GCN层的输出维度
        self.weight_size_list = [self.initial_embed_size] + self.weight_size

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

        # 协同图卷积层的参数初始化
        all_weights['V_symp'] = tf.Variable(
            initializer([self.initial_embed_size, self.weight_size_list[-1]]),
            name = 'V_symp'
        )
        all_weights['B_symp'] = tf.Variable(
            initializer([1, self.weight_size_list[-1]]),
            name = 'B_symp'
        )
        all_weights['V_herb'] = tf.Variable(
            initializer([self.initial_embed_size, self.weight_size_list[-1]]),
            name = 'V_herb'
        )
        all_weights['B_herb'] = tf.Variable(
            initializer([1, self.weight_size_list[-1]]),
            name = 'B_herb'
        )

        return all_weights

    def _create_placeholders(self):
        """
          等待运行时动态填充的数据
        """
        self.sympt_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_symptoms))
        self.herb_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_herbs))
        self.sympt_set_normalized_mat = tf.placeholder(dtype=np.float32, shape=(None, None))

    def _inference(self):
        # 获取每个样本中的症状集嵌入向量之和然后做平均操作
        self.sympt_set_embeddings_sum = tf.matmul(self.sympt_set_sample_mat, self.sympt_embeddings)
        self.mean_sympt_set_embedding = tf.matmul(self.sympt_set_normalized_mat, self.sympt_set_embeddings_sum)

        # 将获得的平均症状嵌入作为一个综合的综合症嵌入表示
        self.syndrome_embedding = self.mean_sympt_set_embedding

        # 将综合的综合症嵌入表示与草药嵌入矩阵做交互，得到大小为|H|的预测向量
        self.prediction_herb_mat = tf.matmul(self.syndrome_embedding, self.herb_embeddings, transpose_a=False, transpose_b=True)

        with tf.name_scope('loss'):
            self.embed_loss, self.reg_loss = self.create_multilabel_loss(self.prediction_herb_mat,
                                                                     self.herb_set_sample_mat)

            self.total_loss = self.embed_loss + self.reg_loss
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("embed_loss", self.embed_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)


    def _create_smgcn_embed(self):
        """
        1.利用Bipar-GCN模型获取症状嵌入和草药嵌入的综合矩阵表示形式ego_embeddings
        :return:
        """
        # 获取最初的症状-草药嵌入矩阵
        ego_embeddings = tf.concat([self.weights['sympt_embeddings'],
                                    self.weights['herb_embeddings']], axis=0)

        """
        2.利用协同图SGE获取另一种草药嵌入和症状嵌入的综合矩阵表示形式
        """
        # 获取草药-草药共现矩阵和症状-症状共现矩阵
        sympt_sympt_cocur_mat = self.data_generator.get_adj_mat()[1]
        herb_herb_cocur_mat = self.data_generator.get_adj_mat()[2]
        # 获取草药-草药邻接矩阵和症状-症状邻接矩阵
        sympt_sympt_adj_mat = self.get_adj_mat_by_threshold(sympt_sympt_cocur_mat, self.args.sympt_threshold)
        herb_herb_adj_mat = self.get_adj_mat_by_threshold(herb_herb_cocur_mat, self.args.herb_threshold)

        # 首先获得症状的嵌入矩阵
        sympt_synergy_embed_mat = tf.nn.tanh(tf.matmul( tf.matmul(sympt_sympt_adj_mat,
                                                                  self.weights['sympt_embeddings']), self.weights['V_symp'])  +
                                             self.weights['B_symp']   )
        # 再获得草药的嵌入矩阵
        herb_synergy_embed_mat = tf.nn.tanh(tf.matmul(tf.matmul(herb_herb_adj_mat,
                                                                self.weights['herb_embeddings']),self.weights['V_herb']) +
                                            self.weights['B_herb'])
        # 将两个嵌入矩阵堆在一起
        synergy_embed_mat = tf.concat([sympt_synergy_embed_mat, herb_synergy_embed_mat], axis=0)

        fused_embedding = synergy_embed_mat

        """
        4.将获得的嵌入矩阵划分为症状嵌入矩阵和草药嵌入矩阵
        """
        sympt_embeddings, herb_embeddings = tf.split(fused_embedding, [self.n_symptoms, self.n_herbs],
                                                     axis=0
                                                     )
        return sympt_embeddings, herb_embeddings


    def _train(self):
        self.sympt_embeddings, self.herb_embeddings = self._create_smgcn_embed()
        self._create_placeholders()
        self._inference()
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.lr,
            global_step=global_step,
            decay_steps=100,
            decay_rate=0.96, 
            staircase=True, 
            name='decayed_lr',
        )
        with tf.name_scope('Adam'):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.total_loss,
                                                                          global_step=global_step)


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

