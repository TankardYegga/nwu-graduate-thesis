import collections
import numpy as np
# 获得目标节点数据
def _compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size, max_node_id):
    # 对邻居序列采样
    def sample(ns):
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    # 邻居序列向量化，得到邻接向量
    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # 对邻居采样，得到邻接矩阵
    adj_mat_full = np.stack(
        [vectorize(sample(neigh_dict[n])) for n in dst_nodes])
    # 标记哪些列非零，后面用于压缩矩阵
    nonzero_cols_mask = np.any(adj_mat_full.astype(np.bool), axis=0)

    # 压缩矩阵：取出不全为零的列
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    # 按行求和
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
    # 按行归一化
    dif_mat = adj_mat / adj_mat_sum

    # 得到所有目标节点的邻接序号
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    # 将目标节点与邻接节点取并集，并且升序排序
    dstsrc = np.union1d(dst_nodes, src_nodes)
    # 标记哪些节点是邻接节点
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    # 标记哪些节点是目标节点
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


# 根据节点构造mini-batch数据
def build_batch_from_nodes(nodes, neigh_dict, sample_sizes):
    """
    参数：
    nodes: 目标节点列表
    neigh_dict: 邻居节点列表
    sample_sizes: 每层采样的个数
    """
    # dst_nodes 实际上是栈，存储了0,1,2...,K阶(邻居)节点集合
    dst_nodes = [nodes]
    dstsrc2dsts = []
    dstsrc2srcs = []
    dif_mats = []

    max_node_id = max(list(neigh_dict.keys()))
    """
    以下是mini-batch采样算法,这里以K层为例，说明一下采样顺序与dst_nodes栈内的数据：
    采样顺序是从K，K-1，... 1:
    B_K(栈底元素): 输入目标节点集合nodes；
    B_K-1: 目标节点+其一阶邻居节点；
    B_K-2: 目标节点+其一阶邻居节点+其二阶邻居节点；
    ...
    B_0(栈顶元素): 目标节点+其一阶邻居节点+二阶邻居节点+...+K阶邻居节点。
    """
    for sample_size in reversed(sample_sizes):
        # _compute_diffusion_matrix：
        # 对目标节点dst_nodes[-1]邻居采样sample_size个
        # 当dst_nodes[-1]==nodes时，需要对nodes的邻居
        # ds 是目标节点、邻居节点并集
        # d2s 是ds中邻居节点的序号
        # d2d 是ds中目标节点的序号
        ds, d2s, d2d, dm = _compute_diffusion_matrix(dst_nodes[-1]
                                                     , neigh_dict
                                                     , sample_size
                                                     , max_node_id
                                                     )
        dst_nodes.append(ds)
        dstsrc2srcs.append(d2s)
        dstsrc2dsts.append(d2d)
        dif_mats.append(dm)

    src_nodes = dst_nodes.pop()

    MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
    MiniBatch = collections.namedtuple("MiniBatch", MiniBatchFields)

    return MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats)
