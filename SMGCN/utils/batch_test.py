#coding=utf-8
"""
  用于pinsage-pytorch的test
"""
import math
import multiprocessing
import os
import  numpy as np
import  heapq

#返回的是计算机的cpu核心数，比如cpu是双核的，则返回2，如果双四核cpu，则返回8
from utils.metrics import  cal_auc, precision_at_k, ndcg_at_k, dcg_at_k, hit_at_k, recall_at_k

tested_case_count = 0

"""
测试模型
"""
def test_model2(sess, model,  data_generator, args):
    global tested_case_count
    tested_case_count = 0
    len_Ks = len(eval(args.Ks))
    test_result = {'precision': np.zeros(len_Ks), 'recall': np.zeros(len_Ks),
                   'ndcg': np.zeros(len_Ks), 'hit_ratio': np.zeros(len_Ks)}
    #开启并发线程池
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)

    # 将测试集的处方按照batch_size进行划分
    n_pres_batch = math.ceil(data_generator.n_train/args.batch_size )

    count = 0
    test_loss = 0.0

    # 计算该batch_size个处方对应的预测结果
    for i_batch in range(n_pres_batch):
        # 获取此batch_size个处方
        pres_start_idx = i_batch * args.batch_size
        pres_end_idx = min((i_batch+1) * args.batch_size, data_generator.n_train)
        batch_pres = data_generator.train_prescrp_list[pres_start_idx:pres_end_idx]

        # 构造这args.batch_size个处方对应的症状矩阵、草药矩阵
        # 矩阵每一行是该处方中草药集或者症状集的one-hot向量
        batch_sympt_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), data_generator.n_symptoms))
        batch_herb_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), data_generator.n_herbs))
        batch_sympt_set_normalized_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres),))

        for pres_idx in range(len(batch_pres)):
            single_pres = batch_pres[pres_idx]
            single_sympt_set = single_pres.symptoms
            single_herb_set = single_pres.herbs
            for sympt in single_sympt_set:
                batch_sympt_set_mat[int(pres_idx)][int(sympt)] = 1
            for herb in single_herb_set:
                batch_herb_set_mat[int(pres_idx)][int(herb)] = 1
            batch_sympt_set_normalized_mat[pres_idx] = len(single_sympt_set) if len(single_sympt_set)!=0 else 1
        batch_sympt_set_normalized_mat = np.diag(np.power(batch_sympt_set_normalized_mat,-1))

        # 将这三个矩阵馈入到模型当中，可以得到损失以及推荐的草药集
        test_batch_total_loss, prediction_herb_mat = sess.run([model.total_loss,
                                            model.prediction_herb_mat],
                                            feed_dict={ model.sympt_set_sample_mat:batch_sympt_set_mat,
                                                      model.herb_set_sample_mat:batch_herb_set_mat,
                                                    model.sympt_set_normalized_mat:batch_sympt_set_normalized_mat
                                                    })
        # if i_batch ==0:
        #     print('type ', type(prediction_herb_mat))
        #     print('p'*100)
        #     print(prediction_herb_mat)
        #     print('p'*100)
        #     print('/'*100)
        #     num_keys = list(range(data_generator.n_herbs))
        #     dict1 = dict(zip(num_keys, list(prediction_herb_mat[0])))
        #     sorted_dict = sorted(dict1, reverse=True,key=dict1.get)
        #     print(sorted_dict[:20])
        #     print('/'*100)
        #     print('9'*100)
        #     for idx in range(len(batch_herb_set_mat[0])):
        #         x = batch_herb_set_mat[0][idx]
        #         if x!=0:
        #             print('%d:%d,'% (idx,x), end=' ')
        #     print('9'*100)
        test_loss += test_batch_total_loss
        predict_herbs_vs_true_herbs = zip(prediction_herb_mat, batch_herb_set_mat)
        # batch_test_res = pool.map(test_one_prescrp, predict_herbs_vs_true_herbs)
        batch_test_res = []
        for x in predict_herbs_vs_true_herbs:
            x_res = test_one_prescrp(x)
            batch_test_res.append(x_res)
        count += len(batch_test_res)

        for res in batch_test_res:
            test_result['precision'] += res['precision']
            test_result['recall'] += res['recall']
            test_result['ndcg'] += res['ndcg']
            test_result['hit_ratio'] += res['hit_ratio']

    assert count == data_generator.n_train
    test_result['precision'] /= data_generator.n_train
    test_result['recall'] /= data_generator.n_train
    test_result['ndcg'] /= data_generator.n_train
    test_result['hit_ratio'] /= data_generator.n_train
    pool.close()
    return test_result, test_loss


# 计算该batch_size个处方中每个处方预测结果的测试指标
def test_one_prescrp(x,  data_generator, args):
    global  tested_case_count
    tested_case_count=tested_case_count+1
    # 获取预测草药集
    predict_herbs = x[0]
    # 获取真实草药集
    true_herbs = x[1]

    # 计算统计结果auc
    auc = cal_auc(x[1], x[0])

    # 将predict_herbs转化成字典
    num_keys = list(range(data_generator.n_herbs))
    predict_herbs_dict = dict(zip(num_keys, predict_herbs))

    # 构建关系向量r，用来表示预测的草药集与真实草药集合的关系
    # 即是：若预测的草药在真实草药集中，则对应项标记为1，否则为0
    # 这里取K的所有可能取值中的最大值
    maxk = max(eval(args.Ks))
    k_max_herbs_score = heapq.nlargest(maxk, predict_herbs_dict, key=predict_herbs_dict.get)

    r = []
    # print('k_max', k_max_herbs_score)
    for predict_herb in k_max_herbs_score:
        if true_herbs[predict_herb] != 0:
            r.append(1)
        else:
            r.append(0)

    # print('1'*80)
    # print('The r is', r)
    # inter_sess = tf.InteractiveSession()
    # print('cross similarity is ', inter_sess.run(tf.reduce_sum(tf.multiply(x[0], x[1]))))
    # print('predict_herbs[times=%d, len=%d]:'%(tested_case_count, len(k_max_herbs_score)), k_max_herbs_score)
    true_herbs_set = [x!=0 for x in true_herbs]
    true_herbs_output = []
    for index in range(len(true_herbs_set)):
        if true_herbs_set[index]:
            true_herbs_output.append(index)
    # print('true_herbs[times=%d, len=%d]:'% (tested_case_count, len(true_herbs_output)), true_herbs_output)

    precision, recall, ndcg, hit_ratio = [], [], [], []
    for k in eval(args.Ks):
        precision.append(precision_at_k(r,k))
        recall.append(recall_at_k(r,k, len(true_herbs_output)))
        ndcg.append(ndcg_at_k(r, k))
        hit_ratio.append(hit_at_k(r,k))

    result = {'precision': np.array(precision), 'recall': np.array(recall),
               'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

    # print('precision', result['precision'])
    # print('recall', result['recall'])
    # print('ndcg', result['ndcg'])
    # print('hit_ratio', result['hit_ratio'])
    # print('1'*80, '\n')

    return result


def test_model(sess, model,  data_generator, args):
    global tested_case_count
    tested_case_count = 0
    len_Ks = len(eval(args.Ks))
    test_result = {'precision': np.zeros(len_Ks), 'recall': np.zeros(len_Ks),
                   'ndcg': np.zeros(len_Ks), 'hit_ratio': np.zeros(len_Ks)}
    #开启并发线程池
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)

    # 将测试集的处方按照batch_size进行划分
    n_pres_batch = math.ceil(data_generator.n_test/args.batch_size )

    count = 0
    test_loss = 0.0

    # 计算该batch_size个处方对应的预测结果
    for i_batch in range(n_pres_batch):
        # 获取此batch_size个处方
        pres_start_idx = i_batch * args.batch_size
        pres_end_idx = min((i_batch+1) * args.batch_size, data_generator.n_test)
        batch_pres = data_generator.test_prescrp_list[pres_start_idx:pres_end_idx]

        # 构造这args.batch_size个处方对应的症状矩阵、草药矩阵
        # 矩阵每一行是该处方中草药集或者症状集的one-hot向量
        batch_sympt_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), data_generator.n_symptoms))
        batch_herb_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), data_generator.n_herbs))
        batch_sympt_set_normalized_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres),))


        for pres_idx in range(len(batch_pres)):
            single_pres = batch_pres[pres_idx]
            single_sympt_set = single_pres.symptoms
            single_herb_set = single_pres.herbs
            for sympt in single_sympt_set:
                batch_sympt_set_mat[int(pres_idx)][int(sympt)] = 1
            for herb in single_herb_set:
                batch_herb_set_mat[int(pres_idx)][int(herb)] = 1
            batch_sympt_set_normalized_mat[pres_idx] = len(single_sympt_set) if len(single_sympt_set)!=0 else 1
        batch_sympt_set_normalized_mat = np.diag(np.power(batch_sympt_set_normalized_mat,-1))

        # 将这三个矩阵馈入到模型当中，可以得到损失以及推荐的草药集
        test_batch_total_loss, prediction_herb_mat = sess.run([model.total_loss,
                                                               model.prediction_herb_mat],
                                                              feed_dict={ model.sympt_set_sample_mat:batch_sympt_set_mat,
                                                                          model.herb_set_sample_mat:batch_herb_set_mat,
                                                                          model.sympt_set_normalized_mat:batch_sympt_set_normalized_mat
                                                                          })
        test_loss += test_batch_total_loss
        predict_herbs_vs_true_herbs = zip(prediction_herb_mat, batch_herb_set_mat)
        # batch_test_res = pool.map(test_one_prescrp, predict_herbs_vs_true_herbs)
        batch_test_res = []
        for x in predict_herbs_vs_true_herbs:
            x_res = test_one_prescrp(x, data_generator, args)
            batch_test_res.append(x_res)
        count += len(batch_test_res)

        for res in batch_test_res:
            test_result['precision'] += res['precision']
            test_result['recall'] += res['recall']
            test_result['ndcg'] += res['ndcg']
            test_result['hit_ratio'] += res['hit_ratio']

    assert count == data_generator.n_test
    test_result['precision'] /= data_generator.n_test
    test_result['recall'] /= data_generator.n_test
    test_result['ndcg'] /= data_generator.n_test
    test_result['hit_ratio'] /= data_generator.n_test
    pool.close()
    return test_result, test_loss





