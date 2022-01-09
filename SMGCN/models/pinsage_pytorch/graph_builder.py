import dgl
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


# 构建sympt-herb的异质网络图
class GraphBuilder(object):

    def __init__(self, data_generator, node_adj_mat,):
        self.data_generator = data_generator
        self.node_adj_mat = node_adj_mat
        self.get_nodes_info()
        self.get_edges_info()

    def get_nodes_info(self):
        self.num_nodes_per_type = {}
        self.num_nodes_per_type['sympt'] = self.data_generator.n_symptoms
        self.num_nodes_per_type['herb'] = self.data_generator.n_herbs

    def get_edges_info(self):
        self.edges_per_relation = {}
        # 从sympt-herb的邻接矩阵中去获取相应的下标
        self.sympt_indice = []
        self.herb_indice = []
        for sympt_idx in range(self.data_generator.n_symptoms):
            for herb_idx in range(self.data_generator.n_herbs):
                if self.data_generator.R[sympt_idx, herb_idx]:
                    self.sympt_indice.append(sympt_idx)
                    self.herb_indice.append(herb_idx)
        sympt_indice = torch.LongTensor(self.sympt_indice)
        herb_indice = torch.LongTensor(self.herb_indice)

        self.edges_per_relation[('sympt', 'cured_by', 'herb')] = (sympt_indice, herb_indice)
        self.edges_per_relation[('herb', 'cure', 'sympt')] = (herb_indice, sympt_indice)

    def build(self):
        graph = dgl.heterograph(self.edges_per_relation, self.num_nodes_per_type)
        # nx.draw(graph.to_networkx(), with_label=True)
        # plt.savefig('save/sympt-herb-graph.png')
        # plt.close()
        print('Graph Created ')
        print(graph)
        return graph


class HomoGraphBuilder(object):

    def __init__(self, num_nodes, node_adj_mat, n_type):
        self.n_type = n_type
        self.num_of_nodes = num_nodes
        self.node_adj_mat = node_adj_mat
        self.get_edges_info(node_adj_mat)

    def get_edges_info(self, node_adj_mat):
        self.u = []
        self.v = []
        for i in range(node_adj_mat.shape[0]):
            for j in range(node_adj_mat.shape[1]):
                if node_adj_mat[i][j] != 0:
                    self.u.append(i)
                    self.v.append(j)
        self.src_nodes = torch.LongTensor(self.u)
        self.dst_nodes = torch.LongTensor(self.v)

    def build(self):
        graph = dgl.graph((self.src_nodes, self.dst_nodes), num_nodes=self.num_of_nodes)
        nx.draw(graph.to_networkx(), with_labels=True)
        plt.savefig('save/' + self.n_type + '_homo.png')
        plt.close()
        print('Graph Created ')
        print(graph)
        return graph


    