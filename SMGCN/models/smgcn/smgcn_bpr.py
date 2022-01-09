import sys
# sys.path.append('..')

from models.base_gnn import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SMGCN_BPR(BASE_GNN):
    def __init__(self, **kwargs):
        self.model_type='smgcn'
        super(SMGCN_BPR, self).__init__(**kwargs)
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

        # 维度参数列表：输入维度 + 各个GCN层的输出维度
        self.weight_size_list = [self.initial_embed_size] + self.weight_size

        # 二部图GCN各层的参数初始化
        # 对于草药或者症状，每层都对应两个权重矩阵和其偏置项，
        # 一个是用于构造消息的矩阵T和其偏置项，一个是用于聚合消息的矩阵W和其偏置项
        # 但是对于草药和症状，它们使用的T和W都是不同的
        # 所以每层有2*4=8个参数
        for k in range(self.n_layers):
            all_weights['T_%d_sympt_construt' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.hidden_size[k]]),
                name = 'T_%d_sympt_construt' % k
            )
            all_weights['b_%d_sympt_construct' % k] = tf.Variable(
                initializer([1, self.hidden_size[k]]),
                name = 'b_%d_sympt_construct' % k
            )
            all_weights['W_%d_sympt_aggregate' % k] = tf.Variable(
                initializer([self.hidden_size[k] + self.weight_size_list[k], self.weight_size_list[k+1]]),
                name = 'W_%d_sympt_aggregate' % k
            )
            all_weights['b_%d_sympt_aggregate' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]),
                name = 'b_%d_sympt_aggregate' % k
            )

            all_weights['T_%d_herb_construct' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.hidden_size[k]]),
                name = 'T_%d_herb_construct' % k
            )
            all_weights['b_%d_herb_construct' % k] = tf.Variable(
                initializer([1, self.hidden_size[k]]),
                name = 'b_%d_herb_construct' % k
            )
            all_weights['W_%d_herb_aggregate' % k] = tf.Variable(
                initializer([self.hidden_size[k] + self.weight_size_list[k], self.weight_size_list[k+1]]),
                name = 'W_%d_herb_aggregate' % k
            )
            all_weights['b_%d_herb_aggregate' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]),
                name = 'b_%d_herb_aggregate' % k
            )

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

        # MLP层卷积层的参数初始化
        all_weights['MLP_W'] = tf.Variable(
            initializer([self.weight_size_list[-1], self.weight_size_list[-1]]),
            name='MLP_W'
        )
        all_weights['MLP_b'] = tf.Variable(
            initializer([1, self.weight_size_list[-1]]),
            name='MLP_b'
        )

        return all_weights


    def _create_placeholders(self):
        """
          等待运行时动态填充的数据
        """
        # self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        # self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        # placeholder definition
        self.sympt_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_symptoms))
        self.sympt_set_normalized_mat = tf.placeholder(dtype=np.float32, shape=(None, None))
        self.pos_herbs = tf.placeholder(tf.int32, shape=(None,))
        self.neg_herbs = tf.placeholder(tf.int32, shape=(None,))
        # self.herb_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_herbs))


    def _inference(self):
        # 获取每个样本中的症状集嵌入向量之和然后做平均操作
        self.sympt_set_embeddings_sum = tf.matmul(self.sympt_set_sample_mat, self.sympt_embeddings)
        self.mean_sympt_set_embedding = tf.matmul(self.sympt_set_normalized_mat, self.sympt_set_embeddings_sum)

        # 将获得的平均症状嵌入通过MLP获得一个综合的综合症嵌入表示
        self.syndrome_embedding = tf.nn.relu(tf.matmul(self.mean_sympt_set_embedding, self.weights['MLP_W']) + \
                                             self.weights['MLP_b'])
        
        # 将综合的综合症嵌入表示与草药嵌入矩阵做交互，得到大小为|H|的预测向量
        self.prediction_herb_mat = tf.matmul(self.syndrome_embedding, self.herb_embeddings, 
            transpose_a=False, transpose_b=True)

        self.pos_herb_embeddings = tf.nn.embedding_lookup(self.herb_embeddings, self.pos_herbs)
        self.neg_herb_embeddings = tf.nn.embedding_lookup(self.herb_embeddings, self.neg_herbs)

        with tf.name_scope('loss'):
            self.embed_loss, self.reg_loss = self.create_bpr_loss(self.syndrome_embedding,
                                                              self.pos_herb_embeddings,
                                                              self.neg_herb_embeddings)

            self.total_loss = self.embed_loss + self.reg_loss
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("embed_loss", self.embed_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)


    def _create_smgcn_bpr_embed(self):
        """
        1.利用Bipar-GCN模型获取症状嵌入和草药嵌入的综合矩阵表示形式ego_embeddings
        :return:
        """
        # 获取最初的症状-草药嵌入矩阵
        ego_embeddings = tf.concat([self.weights['sympt_embeddings'],
                                    self.weights['herb_embeddings']], axis=0)
        A = tf.convert_to_tensor(self.data_generator.A)
        # 求A对应的度数矩阵
        D = np.sum(self.data_generator.A, axis=1, keepdims=False) + 1
        D = tf.convert_to_tensor(np.diag(np.power(D,-1)))
        for k in range(0,self.n_layers):
            # 获取每个结点嵌入对应周围邻居嵌入的和
            side_embeddings = tf.matmul(A, ego_embeddings)
            syndrome_neighbor_msg = tf.matmul(side_embeddings[:self.n_symptoms],
                                              self.weights['T_%d_sympt_construt' % k]) \
                                    + self.weights['b_%d_sympt_construct' % k]
            herb_neighbor_msg = tf.matmul(side_embeddings[self.n_symptoms:],
                                          self.weights['T_%d_herb_construct' % k]) + \
                                self.weights['b_%d_herb_construct' % k]
            # merged neigbor msg
            neigbor_msg_mat = tf.concat([syndrome_neighbor_msg, herb_neighbor_msg], 0)
            normalized_neigbor_msg_mat = tf.matmul(D, neigbor_msg_mat)
            neighbor_info_mat = tf.nn.tanh(normalized_neigbor_msg_mat)

            # aggregated syndrome embed matrix
            aggregated_syndrome_msg = tf.nn.tanh(tf.matmul(
                tf.concat([neighbor_info_mat[:self.n_symptoms], ego_embeddings[:self.n_symptoms]],axis=1),
                self.weights['W_%d_sympt_aggregate' % k]
            ) + self.weights['b_%d_sympt_aggregate' % k])
            # aggregated herb embed matrix
            aggregated_herb_msg = tf.nn.tanh(tf.matmul(
                tf.concat([neighbor_info_mat[self.n_symptoms:], ego_embeddings[self.n_symptoms:]],axis=1),
                self.weights['W_%d_herb_aggregate' % k]
            ) + self.weights['b_%d_herb_aggregate' % k])

            # 更新嵌入矩阵
            ego_embeddings = tf.concat([aggregated_syndrome_msg, aggregated_herb_msg],axis=0)

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
                                             self.weights['B_symp']      )
        # 再获得草药的嵌入矩阵
        herb_synergy_embed_mat = tf.nn.tanh(tf.matmul(tf.matmul(herb_herb_adj_mat,
                                                                self.weights['herb_embeddings']),self.weights['V_herb']) +
                                            self.weights['B_herb'])
        # 将两个嵌入矩阵堆在一起
        synergy_embed_mat = tf.concat([sympt_synergy_embed_mat, herb_synergy_embed_mat], axis=0)

        """
        3.通过相加来融合两种嵌入表示形式
        """
        fused_embedding = tf.add(ego_embeddings, synergy_embed_mat)

        """
        4.将获得的嵌入矩阵划分为症状嵌入矩阵和草药嵌入矩阵
        """
        sympt_embeddings, herb_embeddings = tf.split(fused_embedding, [self.n_symptoms, self.n_herbs],
                                                     axis=0
                                                     )
        return sympt_embeddings, herb_embeddings


    def _train(self):
        self.sympt_embeddings, self.herb_embeddings = self._create_smgcn_bpr_embed()
        self._create_placeholders()
        self._inference()
        
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope('Adam'):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss,
                                                                          global_step=global_step)


    def get_adj_mat_by_threshold(self, cocur_mat, threshold):
        adj_mat = np.zeros(dtype=np.float32, shape=cocur_mat.shape)
        for i in range(cocur_mat.shape[0]):
            for j in range(i+1, cocur_mat.shape[1]):
                if cocur_mat[i][j] >= threshold:
                    adj_mat[i][j] = 1.0
                    adj_mat[j][i] = 1.0
        return adj_mat

    
    # 此仍旧采用一个正样本和一个负样本的形式
    # 对于一个给定处方中的症状集，其正样本对应于处方中草药集中的任意一个草药
    # 其负样本对应于 （全部草药 - 处方中草药集）中的任何一个草药
    def create_bpr_loss(self, mean_sympt_set_embeddings, pos_herb_embeddings, neg_herb_embeddings):
        pos_scores = tf.reduce_sum(tf.multiply(mean_sympt_set_embeddings, pos_herb_embeddings), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(mean_sympt_set_embeddings, neg_herb_embeddings), axis=1)

        regularizer = tf.nn.l2_loss(mean_sympt_set_embeddings) + tf.nn.l2_loss(pos_herb_embeddings) + \
                      tf.nn.l2_loss(neg_herb_embeddings)
        regularizer = regularizer/self.batch_size

        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        embed_loss = tf.negative(tf.reduce_mean(maxi))

        reg_loss = eval(self.args.regs) * regularizer

        return  embed_loss,reg_loss

