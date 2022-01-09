#coding=utf-8
import numpy as np
from sklearn.metrics import roc_auc_score

def cal_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

# 求解准确度
def precision_at_k(r, k):
    # 确保k>=1，是保证起码有1个预测的草药从而r!=[],因为np.mean([])=nan
    assert  k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)  # 正确预测草药数目/总的预测草药数目

# 求解回召率
def recall_at_k(r, k, true_herbs_num):
    r = np.asfarray(r)[:k] # 转化为float
    return np.sum(r) / true_herbs_num # 正确预测草药的数目 / 真实草药集中草药的总数目

# 判断是否命中，即是否有预测正确的草药
def hit_at_k(r, k):
    r = np.asarray(r)[:k]
    if np.sum(r) > 0:
        return 1
    else:
        return 0

# 求解dcg, method控制依据位置折损的方式
def dcg_at_k(r, k, method=1):
    r = np.asarray(r)[:k]
    if r.size:
        if  method == 0:
            return r[0] + np.sum(r[1:]/np.log2(np.arange(2, k+1)))
        elif method == 1:
            return np.sum( r / np.log2(np.arange(2, k+2)) )
        else:
            raise ValueError('Method must be 0 or 1')
    return 0.

# 求解ndcg
def ndcg_at_k(r,k):
    # 求解idcg
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, 1)
    if not dcg_max:
        return 0.
    return dcg_at_k(r,k,1) / dcg_max







