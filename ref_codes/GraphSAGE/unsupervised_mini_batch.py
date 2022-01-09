import collections
import numpy as np

from GraphSAGE.supervised_mini_batch import build_batch_from_nodes


def _get_neighbors(nodes, neigh_dict):
    return np.unique(np.concatenate([neigh_dict[n] for n in nodes]))

# 无监督学习时，根据边得到 mini-batch 数据
def build_batch_from_edges(edges, nodes, neigh_dict, sample_sizes, neg_size):
    # batchA 目标节点列表
    # batchB 与目标节点对应的邻居节点列表
    batchA, batchB = edges.transpose()
    # 从 nodes 中去除 batchA、batchA节点邻居，batchB、batchB节点邻居
    # 执行过程：((((nodes-batchA)-neighbor_A)-batchB) - neighbor_B)
    # 得到所有可能的负样本
    possible_negs = reduce ( np.setdiff1d
                           , ( nodes
                             , batchA
                             , _get_neighbors(batchA, neigh_dict)
                             , batchB
                             , _get_neighbors(batchB, neigh_dict)
                             )
                           )
    # 从所有负样本中采样出neg_size个
    batchN = np.random.choice ( possible_negs
                              , min(neg_size, len(possible_negs))
                              , replace=False
                              )

    # np.unique：去重，结果已排序
    batch_all = np.unique(np.concatenate((batchA, batchB, batchN)))
    # 得到batchA、batchB在batch_all中的序号
    dst2batchA = np.searchsorted(batch_all, batchA)
    dst2batchB = np.searchsorted(batch_all, batchB)
    # 计算batch_all每个元素在batchN中是否出现
    dst2batchN = np.in1d(batch_all, batchN)
    # 上面已经完成了边的采样，并且得到边的节点
    # 接下来是构造mini-batch数据
    minibatch_plain = build_batch_from_nodes ( batch_all
                                             , neigh_dict
                                             , sample_sizes
                                             )

    MiniBatchFields = [ "src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"
                      , "dst2batchA", "dst2batchB", "dst2batchN" ]
    MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)

    return MiniBatch ( minibatch_plain.src_nodes # 目标节点与邻居节点集合
                     , minibatch_plain.dstsrc2srcs # 邻居节点集合
                     , minibatch_plain.dstsrc2dsts # 目标节点集合
                     , minibatch_plain.dif_mats # 归一化矩阵
                     , dst2batchA # 随机采样边的左顶点
                     , dst2batchB # 随机采样边的右顶点
                     , dst2batchN # 标记是否为负采样节点的mask
                     )
