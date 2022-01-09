import tensorflow as tf
from models.base_gnn import *
from utils.mat_operation import normalized_adj_single
import scipy.sparse as sp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NGCF(BASE_GNN):
    def __init__(self, **kwargs):
        super(NGCF, self).__init__(**kwargs)
        self.adj_type = self.args.adj_type
        self.model_type = '_%s_%s_l%d' % (self.adj_type, self.args.alg_type, self.n_layers)

        self.n_fold = 100

        self.norm_adj = normalized_adj_single(self.data_generator.A, method=2)
        self.added_norm_adj = self.norm_adj  + sp.eye(self.norm_adj.shape[0]) 
        
        self._train()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['sympt_embedding'] = tf.Variable(initializer([self.n_symptoms, self.initial_embed_size]), name='sympt_embedding')
            all_weights['herb_embedding'] = tf.Variable(initializer([self.n_herbs, self.initial_embed_size]), name='herb_embedding')
            print('using xavier initialization')
        else:
            all_weights['sympt_embedding'] = tf.Variable(initial_value=self.pretrain_data['sympt_embedding'], trainable=True,
                                                         name='sympt_embedding', dtype=tf.float32)
            all_weights['herb_embedding'] = tf.Variable(initial_value=self.pretrain_data['herb_embedding'], trainable=True,
                                                        name='herb_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.initial_embed_size] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

        # concated_size = sum(self.weight_size_list)
        # print('The concated_size is', concated_size)
        # # MLP层卷积层的参数初始化
        # all_weights['MLP_W'] = tf.Variable(
        #     initializer([ concated_size, concated_size ]),
        #     name='MLP_W'
        # )
        # all_weights['MLP_b'] = tf.Variable(
        #     initializer([1, concated_size ]),
        #     name='MLP_b'
        # )

        return all_weights


    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.added_norm_adj)
            A_fold_hat_2 = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.added_norm_adj)
            A_fold_hat_2 = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['sympt_embedding'], self.weights['herb_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            temp_embed2 = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
                temp_embed2.append(tf.sparse_tensor_dense_matmul(A_fold_hat_2[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            side_embeddings2 = tf.concat(temp_embed2, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(side_embeddings2, ego_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        sympt_embeddings, herb_embeddings = tf.split(all_embeddings, [self.n_symptoms, self.n_herbs], 0)
        return sympt_embeddings, herb_embeddings


    def _create_placeholders(self):
        # placeholder definition
        self.sympt_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_symptoms))
        self.herb_set_sample_mat = tf.placeholder(dtype=np.float32, shape=(None, self.n_herbs))
        self.sympt_set_normalized_mat = tf.placeholder(dtype=np.float32, shape=(None, None))

    def _inference(self):
        # 获取每个样本中的症状集嵌入向量之和然后做平均操作
        self.sympt_set_embeddings_sum = tf.matmul(self.sympt_set_sample_mat, self.sympt_embeddings)
        self.mean_sympt_set_embedding = tf.matmul(self.sympt_set_normalized_mat, self.sympt_set_embeddings_sum)

        # # 将获得的平均症状嵌入通过MLP获得一个综合的综合症嵌入表示
        # self.syndrome_embedding = tf.nn.relu(tf.matmul(self.mean_sympt_set_embedding, self.weights['MLP_W']) + \
        #                                      self.weights['MLP_b'])
        self.syndrome_embedding = self.mean_sympt_set_embedding

        # 将综合的综合症嵌入表示与草药嵌入矩阵做交互，得到大小为|H|的预测向量
        self.prediction_herb_mat = tf.matmul(self.syndrome_embedding, self.herb_embeddings, transpose_a=False, transpose_b=True)

        with tf.name_scope('loss'):
            self.embed_loss, self.reg_loss = self.create_multilabel_loss(self.prediction_herb_mat,
                                                                        self.herb_set_sample_mat)

            self.total_loss = self.embed_loss + self.reg_loss
            tf.summary.scalar("total_loss", self.total_loss)


    def _train(self):
        self.sympt_embeddings, self.herb_embeddings = self._create_ngcf_embed()
        self._create_placeholders()
        self._inference()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

    def _split_A_hat(self, X):
        A_fold_hat = []

        X_copy = sp.csr_matrix(X)

        fold_len = (self.n_symptoms + self.n_herbs) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_symptoms + self.n_herbs
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X_copy[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        X_copy = sp.csr_matrix(X)

        fold_len = (self.n_symptoms + self.n_herbs) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_symptoms + self.n_herbs
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X_copy[start:end])
            n_nonzero_temp = X_copy[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

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
        regularizer = tf.contrib.layers.l2_regularizer(self.decay)
        reg_loss = 0.0
        for param in self.weights.values():
            reg_loss += regularizer(param)

        return embed_loss, reg_loss / self.batch_size


