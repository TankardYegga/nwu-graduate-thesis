# coding=utf-8
import  argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GCMC")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--dataset', nargs='?', default='TCM1',
                          help="Choose a dataset from {TCM1}")
    parser.add_argument('--data_path', nargs='?', default='datasets/TCM1',
                        help='Input data path.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Num of epoch')
    parser.add_argument('--pretrain', type=int, default=0,
                       help='0:No pretrain, -1:Pretrain with learned embeddings, 1:\
                         Pretrain with stored model')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 , 1 ')

    # 初始嵌入维度固定为64
    parser.add_argument('--initial_embed_size', type=int, default=64,
                        help='Initial embedding size for both symptom and herb.')
    # 初始嵌入维度固定为64
    parser.add_argument('--hidden-size', nargs='?', default='[64,128]')
     # 第一层的输出维度默认为128，最后一层的输出维度在{64,128,256,512}中选择
    parser.add_argument('---weight_size', nargs='?', default='[128,256]',
                        help='Output size for every layer')

    # 草药-草药和症状-症状协同图的阈值设定
    parser.add_argument('--sympt_threshold', type=int, default=5,
                        help='Threshold for symptom cocurrence')
    parser.add_argument('--herb_threshold', type=int, default=40,
                        help='Threshold for herb cocurrence.')

    # verbose:迭代训练时的评估间隔，即每verbose次迭代输出一次迭代的相关统计结果
    # 通常默认为1,即每次迭代都输出结果
    parser.add_argument('--verbose', type=int, default=1, help='Interval of outputting evaluation for training.')

    # k的取值范围
    parser.add_argument('--Ks', nargs='?', default='[5,10,20]',
                        help='The value range of K')
    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.0, 0.0, 0.0]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio)'
                             ' for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.0, 0.0, 0.0]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    # 批处理的大小
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    # 学习率
    parser.add_argument('--lr', type=float, default=9e-4,
                        help='Learning rate.')
    # 正则项
    parser.add_argument('--regs', nargs='?', default='[1e-6]',
                        help='Regularizations.')

    parser.add_argument('--model_type', nargs='?', default='gcmc',
                        help='Specify the name of model.')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

    return parser.parse_args()