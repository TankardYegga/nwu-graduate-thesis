import networkx as nx
import decorator
print(decorator.__version__)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
from models.pinsage_pytorch.graph_builder import HomoGraphBuilder
import csv

def visual_homo_graph(graph, graph_name, node_color, csv_name, entity_list):
    G = nx.MultiGraph(with_labels=True)
    for node in range(graph.num_of_nodes):
        G.add_node(str(node))
    
    f = open(csv_name, "w", encoding='utf-8',newline='')
    csv_witer = csv.writer(f)
    csv_witer.writerow(['Source','Target','Weight'])
    edges_list = []
    for elem in zip(graph.u, graph.v):
        w = graph.node_adj_mat[elem[0]][elem[1]]
        t = (elem[0],elem[1],w)
        csv_witer.writerow([entity_list[elem[0]], entity_list[elem[1]], w])
        print(t)
        edges_list.append(t)
    f.close()
    G.add_weighted_edges_from((edges_list))

    nx.draw(G, pos=nx.random_layout(G), node_color = node_color,edge_color = 'gray',with_labels = True, width=0.3, alpha=0.6, 
                node_size=150,  font_size=5, font_color='black')
    # plt.figure(figsize=(10, 10))
    # print(decorator.__version__)
    plt.savefig('extra/' + graph_name, bbox_inches = 'tight')


def visual_hete_graph(graph, graph_name, node_color,  csv_name, entity_list):
    G = nx.MultiDiGraph(with_labels=True)
    num_nodes = graph.num_nodes_per_type['sympt'] + graph.num_nodes_per_type['herb']
    for node in range(num_nodes):
        G.add_node(str(node))
    
    edges_list = []
    for elem in zip(graph.sympt_indice, graph.herb_indice):
        w = graph.node_adj_mat[elem[0]][elem[1]]
        t = (elem[0], elem[1], w)
        print(t)
        edges_list.append(t)
    G.add_weighted_edges_from((edges_list))

    nx.draw(G, pos=nx.random_layout(G), node_color = node_color,edge_color = 'gray',with_labels = True, width=0.3, alpha=0.6, 
                node_size=150,  font_size=5, font_color='black')
    # plt.figure(figsize=(10, 10))
    # print(decorator.__version__)
    plt.savefig('extra/' + graph_name, bbox_inches = 'tight')
    

if __name__ == "__main__":
    import torch
    from utils.load_data import Data
    from models.pinsage_pytorch.pinsage_parser import parse_args
    from models.pinsage_pytorch.graph_builder import GraphBuilder
    from models.pinsage_pytorch.pinsage import pinsage_train
   
    args = parse_args()

    print('path is', args.data_path)
    print('Total Epochs is', args.epoch)

    data_generator = Data(path=args.data_path, batch_size=args.batch_size)
    print('The data has been obtained!')

    # sympt_graph = HomoGraphBuilder(data_generator.n_symptoms, data_generator.get_adj_mat()[1], 
    #                   'sympt')
   
    # visual_homo_graph(sympt_graph, 'sympt_nice_graph.png', 'r', 'sympt_edges.csv', data_generator.symptom_list)

    # herb_graph = HomoGraphBuilder(data_generator.n_herbs, data_generator.get_adj_mat()[2], 
    #                 'herb')
    # visual_homo_graph(herb_graph, 'herb_nice_graph.png', 'g', 'herb_edges.csv', data_generator.herb_list)

    # 利用dgl帮助创建一个我们所需的sympt-herb图结构
    sympt_herb_graph = GraphBuilder(data_generator, data_generator.create_bipar_adj_mat()[1])
    visual_hete_graph(sympt_herb_graph,'sympt_herb_graph.png','b')

    # # 使用节点的特征来作为它们的一个属性（即为每个实体学习一个独立可训练的嵌入表示
    # sympt_herb_graph.nodes['sympt'].data['id'] = torch.arange(data_generator.n_symptoms)
    # sympt_herb_graph.nodes['herb'].data['id'] = torch.arange(data_generator.n_herbs)

   
