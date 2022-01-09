

from models.pinsage_pytorch.graph_builder import HomoGraphBuilder


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

    # sympt_graph = HomoGraphBuilder(data_generator.n_symptoms, data_generator.get_adj_mat()[1], 'sympt')
    
    # herb_graph = HomoGraphBuilder(data_generator.n_herbs, data_generator.get_adj_mat()[2], 'herb')


    # 利用dgl帮助创建一个我们所需的sympt-herb图结构
    sympt_herb_graph = GraphBuilder(data_generator, data_generator.create_bipar_adj_mat()[1]).build()

    # 使用节点的特征来作为它们的一个属性（即为每个实体学习一个独立可训练的嵌入表示
    sympt_herb_graph.nodes['sympt'].data['id'] = torch.arange(data_generator.n_symptoms)
    sympt_herb_graph.nodes['herb'].data['id'] = torch.arange(data_generator.n_herbs)

    pinsage_train(data_generator, args, sympt_herb_graph)

