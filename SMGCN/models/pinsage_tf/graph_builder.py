import dgl
import tensorflow as tf  
import tensorflow.compat.v1 as tf1
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


# 构建sympt-herb的异质网络图
class GraphBuilder(object):

    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.get_nodes_info()
        self.get_edges_info()

    def get_nodes_info(self):
        self.num_nodes_per_type = {}
        self.num_nodes_per_type['sympt'] = self.data_generator.n_symptoms
        self.num_nodes_per_type['herb'] = self.data_generator.n_herbs

    def get_edges_info(self):
        self.edges_per_relation = {}
        # 从sympt-herb的邻接矩阵中去获取相应的下标
        sympt_indice = []
        herb_indice = []
        for sympt_idx in range(self.data_generator.n_symptoms):
            for herb_idx in range(self.data_generator.n_herbs):
                if self.data_generator.R[sympt_idx, herb_idx]:
                    sympt_indice.append(sympt_idx)
                    herb_indice.append(herb_idx)
        sympt_indice = tf.convert_to_tensor(sympt_indice)
        herb_indice = tf.convert_to_tensor(herb_indice)

        self.edges_per_relation[('sympt', 'cured_by', 'herb')] = (sympt_indice, herb_indice)
        self.edges_per_relation[('herb', 'cure', 'sympt')] = (herb_indice, sympt_indice)

    def build(self):
        graph = dgl.heterograph(self.edges_per_relation, self.num_nodes_per_type)
        nx.draw(graph.to_networkx(), with_label=True)
        plt.savefig('save/sympt-herb-graph.png')
        plt.close()
        print('Graph Created ')
        print(graph)
        return graph

