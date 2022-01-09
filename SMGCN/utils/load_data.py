#coding=utf-8
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from classes.prescription import *
import random as rd
import math

#获取数据
#包括：1.症状-草药邻接矩阵、症状-症状邻接矩阵、草药-草药邻接矩阵
# 2. 症状数目、草药数目
# 3.

class Data(object):
    def __init__(self, path, batch_size=20):
        self.path = path
        self.batch_size = batch_size

        # 获取症状和样本的个数和列表
        self.symptoms_path = path + '/symptom_contains.txt'
        self.herbs_path = path + '/herbs_contains.txt'
        self.symptom_list, self.n_symptoms = self.get_entity_list(self.symptoms_path)
        self.herb_list, self.n_herbs = self.get_entity_list(self.herbs_path)

        # 分别获取训练集和测试集中的处方列表，并得到训练集和测试集的大小
        self.train_sympt_path = path + '/pre_symptoms_train.txt'
        self.train_herb_path = path + '/pre_herbs_train.txt'
        self.test_sympt_path = path + '/pre_symptoms_test.txt'
        self.test_herb_path = path + '/pre_herbs_test.txt'
        self.train_prescrp_list, self.n_train,self.n_train_interactions = self.get_entity_pair_list(self.train_sympt_path,
                                                                          self.train_herb_path)
        self.test_prescrp_list, self.n_test, self.n_test_interactions = self.get_entity_pair_list(self.test_sympt_path,
                                                                   self.test_herb_path)

        # 获取症状和草药综合的邻接矩阵
        print('self.get_agj_mat type', type(self.get_adj_mat()))
        self.R = self.get_adj_mat()[0]
        print('Type of self.R:', type(self.R))
        A = np.zeros(shape=(self.n_symptoms+self.n_herbs, self.n_symptoms+self.n_herbs),
                               dtype=np.float32)
        A[:self.n_symptoms, self.n_symptoms:] = self.R
        A[self.n_symptoms:, :self.n_symptoms] = self.R.transpose()
        self.A = A


    def get_entity_list(self, file_path):
       entity_list = []
       with open(file_path, 'rb') as f:
           for l in f.readlines():
               if len(l)>0:
                   single_entity = l.decode().strip('\n')
                   entity_list.append(single_entity)
       return entity_list, len(entity_list)

    def get_entity_pair_list(self, sympt_path, herb_path):
        prescrp_list = []
        n_interactions = 0
        # 获取所有处方中的sympt_set
        with open(sympt_path, 'rb') as f:
            sympt_set_list = f.readlines()

        # 获取所有处方中的herb_set
        with open(herb_path, 'rb') as f:
            herb_set_list = f.readlines()

        for idx in range(len(sympt_set_list)):
            cur_sympt_set = sympt_set_list[idx].decode().strip('\n').split(' ')
            cur_herb_set = herb_set_list[idx].decode().strip('\n').split(' ')
            prescrp_list.append(Prescription(cur_sympt_set, cur_herb_set))
            n_interactions += len(cur_herb_set) * len(cur_sympt_set)

        return prescrp_list, len(prescrp_list), n_interactions

    # 构建二部图邻接矩阵
    def create_bipar_adj_mat(self):
        # 构建共现次数矩阵
        sympt_herb_cocur_nums_mat = np.zeros(dtype=np.float32, shape=(self.n_symptoms, self.n_herbs))
        sympt_herb_adj_mat = np.zeros(dtype=np.float32,
                                        shape=(self.n_symptoms, self.n_herbs))
        for single_pres in self.train_prescrp_list:
            sympt_set = [int(x) for x in single_pres.symptoms]
            herb_set = [int(x) for x in single_pres.herbs]
            for sympt_idx in sympt_set:
                for herb_idx in herb_set:
                    sympt_herb_cocur_nums_mat[sympt_idx][herb_idx] += 1.0

        # 转化为0或者1的邻接矩阵
        print('Adj-'*100)
        print('sympt_herb_cocur_mat\n')
        for sympt_idx in range(self.n_symptoms):
            for herb_idx in range(self.n_herbs):
                if sympt_herb_cocur_nums_mat[sympt_idx][herb_idx] != 0:
                    print("sympt %d, herb %d cocours: %d times" % (sympt_idx, herb_idx,
                                        sympt_herb_cocur_nums_mat[sympt_idx][
                                            herb_idx]))
                    sympt_herb_adj_mat[sympt_idx][herb_idx] = 1.0
        print('Adj-'*100)
        return sympt_herb_adj_mat, sympt_herb_cocur_nums_mat


    # 构建同质共现矩阵
    def create_homo_cocur_mat(self, file_path, type):
       # 读取文件，获取所有处方中的症状集或者草药集
       with open(file_path, 'rb') as f:
           entity_set_list = f.readlines()

       # 遍历每一个处方中的症状集或者草药集
       # 构建共现次数矩阵
       mat_size = self.n_symptoms if type == 'sympt' else self.n_herbs
       entity_cocur_nums_mat = np.zeros(dtype=np.float32, shape=(mat_size,mat_size))
       entity_adj_mat = np.zeros(dtype=np.float32, shape=(mat_size,mat_size))
       for entity_set in entity_set_list:
           entity_list = entity_set.decode().strip('\n').split(' ')
           for i in range(len(entity_list)):
               for j in range(i+1, len(entity_list)):
                   entity1 = int(entity_list[i])
                   entity2 = int(entity_list[j])
                   entity_cocur_nums_mat[entity1][entity2] +=1
                   entity_cocur_nums_mat[entity2][entity1] +=1
       # # 根据阈值，将共现次数矩阵转化为元素为0或者1的邻接矩阵
       # for i in range(mat_size):
       #      for j in range(i, mat_size):
       #          is_adj = 1 if entity_cocur_mat[i][j] >= threshold else 0
       #          entity_adj_mat[i][j] = is_adj
       #          entity_adj_mat[j][i] = is_adj
       return entity_cocur_nums_mat

    def get_adj_mat(self):
        try:
            t1 = time()
            sympt_herb_adj_mat = sp.load_npz(self.path + 'sympt_herb_adj_mat.npz')
            sympt_sympt_cocur_nums_mat = sp.load_npz(self.path + 'sympt_sympt_cocur_nums_mat.npz')
            herb_herb_cocur_nums_mat = sp.load_npz(self.path + 'herb_herb_cocur_nums_mat.npz')
            print('already load adj matrix', sympt_herb_adj_mat.shape, time() - t1)
            return sympt_herb_adj_mat.toarray(), sympt_sympt_cocur_nums_mat.toarray(), \
                  herb_herb_cocur_nums_mat.toarray()
        
        except Exception:
            sympt_herb_adj_mat, sympt_herb_cocur_nums_mat = self.create_bipar_adj_mat()
            sympt_sympt_cocur_nums_mat = self.create_homo_cocur_mat(
                self.train_sympt_path, type='sympt')
            herb_herb_cocur_nums_mat = self.create_homo_cocur_mat(
                self.train_herb_path, type='herb')
            sp.save_npz(self.path + 'sympt_herb_adj_mat.npz', sp.csr_matrix(sympt_herb_adj_mat))
            sp.save_npz(self.path + 'sympt_herb_cocur_nums_mat.npz', sp.csr_matrix(sympt_herb_cocur_nums_mat))
            sp.save_npz(self.path + 'sympt_sympt_cocur_nums_mat.npz', sp.csr_matrix(sympt_sympt_cocur_nums_mat))
            sp.save_npz(self.path + 'herb_herb_cocur_nums_mat.npz', sp.csr_matrix(herb_herb_cocur_nums_mat))
            return sympt_herb_adj_mat, sympt_sympt_cocur_nums_mat, herb_herb_cocur_nums_mat

    def get_num_symptoms_herbs(self):
        return self.n_symptoms, self.n_herbs

    # 打印统计数据
    def print_statistics(self):
        print('n_symptoms=%d, n_herbs=%d' % (self.n_symptoms, self.n_herbs))
        print('n_interactions=%d' % (self.n_train_interactions + self.n_test_interactions))
        print('n_train_interactions=%d, n_test_interactions=%d, sparsity=%.5f' % (self.n_train_interactions,
                                                        self.n_test_interactions,
                            (self.n_train_interactions + self.n_test_interactions)/(self.n_symptoms * self.n_herbs)))


    # 从训练集数据中采样batch_size
    def sample_for_train(self, need_unique_sets=False):
        # 分为两种情况
        # 批量的大小要比训练集小，则可以从训练集中随机抽选batch_size个不重复的样本
        if self.batch_size <= self.n_train:
            # random.sample(集合或者是序列，批量大小）
            # self.train_prescrp_list是一个list，pre_sample也是一个list
            pres_sample = rd.sample(self.train_prescrp_list, self.batch_size)
        # 批量的大小要比训练集大，则从训练集中采样batch_size次，样本可能会有重复
        else:
            # random.choice(集合或者序列）即从集合或者序列中选择一个
            pres_sample = [rd.choice(self.train_prescrp_list) for _ in range(self.batch_size)]

        # 将处方样本中分为症状集和草药集两个样本
        sympt_set_sample = [ pres.symptoms for pres in pres_sample]
        herb_set_sample = [pres.herbs for pres in pres_sample]

        # 需要将症状集样本和草药集样本都转化成二维矩阵
        # 其第一维度对应样本数目的多少
        # 其第二维度对应症状集或者草药集的one-hot向量表示
        sympt_set_sample_mat = np.zeros(dtype=np.float32, shape=(self.batch_size, self.n_symptoms))
        herb_set_sample_mat = np.zeros(dtype=np.float32, shape=(self.batch_size, self.n_herbs))
        sympt_set_len_list = np.zeros(dtype=np.float32, shape=(self.batch_size,))

        sympt_seeds = set()
        herb_seeds = set()

        for idx in range(self.batch_size):
            # 取出对应的症状集和草药集
            sympt_set = sympt_set_sample[idx]
            herb_set = herb_set_sample[idx]
            for sympt in sympt_set:
                sympt_set_sample_mat[int(idx)][int(sympt)] = 1
                sympt_seeds.add(sympt)
            for herb in herb_set:
                herb_set_sample_mat[int(idx)][int(herb)] = 1
                herb_seeds.add(herb)

            sympt_set_len_list[idx] = len(sympt_set) if len(sympt_set) !=0 else 1

        # 计算症状集样本所对应矩阵的度矩阵
        sympt_set_normalized_mat = np.diag(np.power(sympt_set_len_list, -1))
        # print('shape is ', sympt_set_normalized_mat.shape)
        if need_unique_sets:
            return sympt_seeds, herb_seeds, sympt_set_sample_mat , herb_set_sample_mat, sympt_set_normalized_mat
        else:
            return sympt_set_sample_mat , herb_set_sample_mat, sympt_set_normalized_mat

    def sample_for_ngcf_bpr(self):
        # 分为两种情况
        # 批量的大小要比训练集小，则可以从训练集中随机抽选batch_size个不重复的样本
        if self.batch_size <= self.n_train:
            # random.sample(集合或者是序列，批量大小）
            # self.train_prescrp_list是一个list，pre_sample也是一个list
            pres_sample = rd.sample(self.train_prescrp_list, self.batch_size)
        # 批量的大小要比训练集大，则从训练集中采样batch_size次，样本可能会有重复
        else:
            # random.choice(集合或者序列）即从集合或者序列中选择一个
            pres_sample = [rd.choice(self.train_prescrp_list) for _ in range(self.batch_size)]

        # 将处方样本中分为症状集和草药集两个样本
        sympt_set_sample = [ pres.symptoms for pres in pres_sample]
        herb_set_sample = [pres.herbs for pres in pres_sample]

        # 为给定的症状集采集正的草药样本
        def sample_pos_herbs_for_sympt_set(sympt_set_idx, num):
            pos_herbs = herb_set_sample[sympt_set_idx]
            n_pos_herbs = len(pos_herbs)
            pos_batch = []

            while True:
                if len(pos_batch) == num: break
                pos_herb_idx = np.random.randint(low=0, high=n_pos_herbs, size=1)[0]
                pos_herb_id = pos_herbs[pos_herb_idx]

                if pos_herb_id not in pos_batch:
                    pos_batch.append(pos_herb_id)
            return pos_batch

        # 为给定的症状集采集负的草药样本
        def sample_neg_herbs_for_sympt_set(sympt_set_idx, num):
            pos_herbs = [int(x) for x in herb_set_sample[sympt_set_idx]]
            neg_herbs = list( set(range(self.n_herbs)) - set(pos_herbs) )
            n_neg_herbs = len(neg_herbs)
            neg_batch = []

            while True:
                if len(neg_batch) == num: break
                neg_herb_idx = np.random.randint(low=0, high=n_neg_herbs, size=1)[0]
                neg_herb_id = neg_herbs[neg_herb_idx]

                if neg_herb_id not in neg_batch:
                    neg_batch.append(neg_herb_id)
            return neg_batch

        pos_herbs, neg_herbs = [], []
        for idx in range(self.batch_size):
            pos_herbs += sample_pos_herbs_for_sympt_set(idx, 1)
            neg_herbs += sample_neg_herbs_for_sympt_set(idx, 1)

        # 需要将症状集样本和草药集样本都转化成二维矩阵
        # 其第一维度对应样本数目的多少
        # 其第二维度对应症状集或者草药集的one-hot向量表示
        sympt_set_sample_mat = np.zeros(dtype=np.float32, shape=(self.batch_size, self.n_symptoms))
        sympt_set_len_list = np.zeros(dtype=np.float32, shape=(self.batch_size,))
        for idx in range(self.batch_size):
            # 取出对应的症状集和草药集
            sympt_set = sympt_set_sample[idx]
            for sympt in sympt_set:
                sympt_set_sample_mat[int(idx)][int(sympt)] = 1
            sympt_set_len_list[idx] = len(sympt_set) if len(sympt_set) != 0 else 1

        # 计算症状集样本所对应矩阵的度矩阵
        sympt_set_normalized_mat = np.diag(np.power(sympt_set_len_list, -1))

        return sympt_set_sample_mat, sympt_set_normalized_mat, pos_herbs, neg_herbs

    
    # 从训练集数据中顺序采样batch_size
    def sample_for_test(self, need_unique_sets=False, i_batch=0):
      
        # 获取此batch_size个处方
        pres_start_idx = i_batch * self.batch_size
        pres_end_idx = min((i_batch+1) * self.batch_size, self.n_test)
        batch_pres = self.test_prescrp_list[pres_start_idx:pres_end_idx]

        # 构造这args.batch_size个处方对应的症状矩阵、草药矩阵
        # 矩阵每一行是该处方中草药集或者症状集的one-hot向量
        batch_sympt_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), self.n_symptoms))
        batch_herb_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), self.n_herbs))
        batch_sympt_set_normalized_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres),))

        sympt_seeds = set()
        herb_seeds = set()

        for pres_idx in range(len(batch_pres)):
            single_pres = batch_pres[pres_idx]
            single_sympt_set = single_pres.symptoms
            single_herb_set = single_pres.herbs
            for sympt in single_sympt_set:
                batch_sympt_set_mat[int(pres_idx)][int(sympt)] = 1
                sympt_seeds.add(sympt)
            for herb in single_herb_set:
                batch_herb_set_mat[int(pres_idx)][int(herb)] = 1
                herb_seeds.add(herb)
            batch_sympt_set_normalized_mat[pres_idx] = len(single_sympt_set) if len(single_sympt_set)!=0 else 1
        batch_sympt_set_normalized_mat = np.diag(np.power(batch_sympt_set_normalized_mat,-1))

        if need_unique_sets:
            return sympt_seeds, herb_seeds, batch_sympt_set_mat, batch_herb_set_mat, batch_sympt_set_normalized_mat
        else:
            return batch_sympt_set_mat, batch_herb_set_mat, batch_sympt_set_normalized_mat


    def sample_for_whole_test(self, need_unique_sets=False):
        # 获取此batch_size个处方
        whole_test_pres = self.test_prescrp_list

        # 构造这args.batch_size个处方对应的症状矩阵、草药矩阵
        # 矩阵每一行是该处方中草药集或者症状集的one-hot向量
        whole_test_sympt_set_mat = np.zeros(dtype=np.float32, shape=(self.n_test, self.n_symptoms))
        whole_test_herb_set_mat = np.zeros(dtype=np.float32, shape=(self.n_test, self.n_herbs))
        whole_test_sympt_set_normalized_mat = np.zeros(dtype=np.float32, shape=(self.n_test,))

        sympt_seeds = set()
        herb_seeds = set()

        for pres_idx in range(self.n_test):
            single_pres = whole_test_pres[pres_idx]
            single_sympt_set = single_pres.symptoms
            single_herb_set = single_pres.herbs

            for sympt in single_sympt_set:
                whole_test_sympt_set_mat[int(pres_idx)][int(sympt)] = 1
                sympt_seeds.add(sympt)
            for herb in single_herb_set:
                whole_test_herb_set_mat[int(pres_idx)][int(herb)] = 1
                herb_seeds.add(herb)

            whole_test_sympt_set_normalized_mat[pres_idx] = len(single_sympt_set) if len(single_sympt_set)!=0 else 1
        whole_test_sympt_set_normalized_mat = np.diag(np.power(whole_test_sympt_set_normalized_mat, -1))

        if need_unique_sets:
            return sympt_seeds, herb_seeds, whole_test_sympt_set_mat, whole_test_herb_set_mat, whole_test_sympt_set_normalized_mat
        else:
            return whole_test_sympt_set_mat, whole_test_herb_set_mat, whole_test_sympt_set_normalized_mat


    def sample_for_whole_train(self, need_unique_sets=False):
         # 获取此batch_size个处方
        whole_train_pres = self.train_prescrp_list

        # 构造这args.batch_size个处方对应的症状矩阵、草药矩阵
        # 矩阵每一行是该处方中草药集或者症状集的one-hot向量
        whole_train_sympt_set_mat = np.zeros(dtype=np.float32, shape=(self.n_train, self.n_symptoms))
        whole_train_herb_set_mat = np.zeros(dtype=np.float32, shape=(self.n_train, self.n_herbs))
        whole_train_sympt_set_normalized_mat = np.zeros(dtype=np.float32, shape=(self.n_train,))

        sympt_seeds = set()
        herb_seeds = set()

        for pres_idx in range(self.n_train):
            single_pres = whole_train_pres[pres_idx]
            single_sympt_set = single_pres.symptoms
            single_herb_set = single_pres.herbs

            for sympt in single_sympt_set:
                whole_train_sympt_set_mat[int(pres_idx)][int(sympt)] = 1
                sympt_seeds.add(sympt)
            for herb in single_herb_set:
                whole_train_herb_set_mat[int(pres_idx)][int(herb)] = 1
                herb_seeds.add(herb)

            whole_train_sympt_set_normalized_mat[pres_idx] = len(single_sympt_set) if len(single_sympt_set)!=0 else 1
        whole_train_sympt_set_normalized_mat = np.diag(np.power(whole_train_sympt_set_normalized_mat, -1))

        if need_unique_sets:
            return sympt_seeds, herb_seeds, whole_train_sympt_set_mat, whole_train_herb_set_mat, whole_train_sympt_set_normalized_mat
        else:
            return whole_train_sympt_set_mat, whole_train_herb_set_mat, whole_train_sympt_set_normalized_mat


    def sample_for_sequential_train(self, need_unique_sets=False, i_batch=0):
        
        # 获取此batch_size个处方
        pres_start_idx = i_batch * self.batch_size
        pres_end_idx = min((i_batch+1) * self.batch_size, self.n_train)
        batch_pres = self.train_prescrp_list[pres_start_idx:pres_end_idx]

        # 构造这args.batch_size个处方对应的症状矩阵、草药矩阵
        # 矩阵每一行是该处方中草药集或者症状集的one-hot向量
        batch_sympt_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), self.n_symptoms))
        batch_herb_set_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres), self.n_herbs))
        batch_sympt_set_normalized_mat = np.zeros(dtype=np.float32, shape=(len(batch_pres),))

        sympt_seeds = set()
        herb_seeds = set()

        for pres_idx in range(len(batch_pres)):
            single_pres = batch_pres[pres_idx]
            single_sympt_set = single_pres.symptoms
            single_herb_set = single_pres.herbs
            for sympt in single_sympt_set:
                batch_sympt_set_mat[int(pres_idx)][int(sympt)] = 1
                sympt_seeds.add(sympt)
            for herb in single_herb_set:
                batch_herb_set_mat[int(pres_idx)][int(herb)] = 1
                herb_seeds.add(herb)
            batch_sympt_set_normalized_mat[pres_idx] = len(single_sympt_set) if len(single_sympt_set)!=0 else 1
        batch_sympt_set_normalized_mat = np.diag(np.power(batch_sympt_set_normalized_mat,-1))

        if need_unique_sets:
            return sympt_seeds, herb_seeds, batch_sympt_set_mat, batch_herb_set_mat, batch_sympt_set_normalized_mat
        else:
            return batch_sympt_set_mat, batch_herb_set_mat, batch_sympt_set_normalized_mat




        



