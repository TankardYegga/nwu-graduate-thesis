import tensorflow as tf
# 平均值聚合器
class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        src_dim: 输入维度
        dst_dim: 输出维度
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight(name=kwargs["name"] + "_weight"
                                 , shape=(src_dim * 2, dst_dim)
                                 , dtype=tf.float32
                                 , initializer=init_fn
                                 , trainable=True
                                 )

    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        """
        dstsrc_features: 第 K-1 层所有节点的 embedding
        dstsrc2dst: 当前层的目标节点
        dstsrc2src: 当前层的邻居节点
        dif_mat: 归一化矩阵
        """
        # 从当前batch所有节点中取出目标节点
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        # 从当前batch所有节点中取出邻居节点
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        # 对邻居节点加权求和，得到邻居节点embedding之和的均值
        # (batch_size, num_neighbors) x (num_neighbors, src_dim)
        aggregated_features = tf.matmul(dif_mat, src_features)
        # 将第k-1层的embedding与聚合结果进行拼接
        concatenated_features = tf.concat([aggregated_features, dst_features],
                                          1)
        # 乘上权重矩阵 w
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)

