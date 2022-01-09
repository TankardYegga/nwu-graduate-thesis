import tensorflow as tf

from GraphSAGE.supervised_graphsage import GraphSageBase
from GraphSAGE.unsupervised_loss_func import compute_uloss


class GraphSageUnsupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, neg_weight):
        super().__init__(raw_features, internal_dim, num_layers, False)
        self.neg_weight = neg_weight

    def call(self, minibatch):
        # 对 embedding 结果进行正则化
        embeddingABN = tf.math.l2_normalize(super().call(minibatch), 1)
        # 损失函数的计算
        self.add_loss (
                compute_uloss ( tf.gather(embeddingABN, minibatch.dst2batchA)
                              , tf.gather(embeddingABN, minibatch.dst2batchB)
                              , tf.boolean_mask(embeddingABN, minibatch.dst2batchN)
                              , self.neg_weight
                              )
                )
        return embeddingABN
