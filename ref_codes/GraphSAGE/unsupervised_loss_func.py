# 无监督学习的损失函数
import tensorflow as tf

@tf.function
def compute_uloss(embeddingA, embeddingB, embeddingN, neg_weight):
    # 计算边的两个节点的内积，得到相似度
    # (batch_size, emb_dim) * (batch_size, emb_dim)
    # -> (batch_size, emb_dim) -> (batch_size, )
    pos_affinity = tf.reduce_sum ( tf.multiply ( embeddingA, embeddingB ), axis=1 )
    # 相当于每个节点都和负样本的 embedding 计算内积，
    # 得到每个节点与每个负样本的相似度
    # (batch_size, emb_dim) x (emb_dim, neg_size) -> (batch_size, neg_size)
    neg_affinity = tf.matmul ( embeddingA, tf.transpose ( embeddingN ) )
    # shape: (batch_size, )
    pos_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.ones_like(pos_affinity)
                                                       , pos_affinity
                                                       , "positive_xent" )
    # shape: (batch_size, neg_num)
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.zeros_like(neg_affinity)
                                                       , neg_affinity
                                                       , "negative_xent" )
    # 对neg_xent所有元素求和后乘上权重
    weighted_neg = tf.multiply ( neg_weight, tf.reduce_sum(neg_xent) )
    # 对两个 loss 进行累加
    batch_loss = tf.add ( tf.reduce_sum(pos_xent), weighted_neg )

    # loss 除以样本个数
    return tf.divide ( batch_loss, embeddingA.shape[0] )
