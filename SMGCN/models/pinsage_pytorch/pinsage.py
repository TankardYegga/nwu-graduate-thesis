

# 原文中的节点嵌入是通过节点相关属性的嵌入进行同纬度映射再相加得到的
# 但是这里应该是随机初始化得到节点的嵌入
# 初始化的嵌入在卷积层上进行传播
# 然后得到更新后的嵌入


# 每次迭代
# 把上一次的嵌入送入下一次的卷积中进行计算？
# 还是说每次都随机初始化呢
# 在想使用tf实现的版本是否在训练过程中也改变了初始的嵌入呢
# 可能论文中pinsage的实现并没有用到dgl库，所以没有出现相应问题
# 损失是怎么计算的呢？
# 原模型中相当于是每次都用相同的方式初始化节点嵌入，所以得到的节点初始嵌入是一样的
# 然后通过模型中相关参数的组合计算得到更新后的所有节点嵌入
# 然后利用所有的节点嵌入在训练集上去计算损失
# 最后根据损失去调整参数

# 所以这里也是一样
# 只需要用固定化方式初始嵌入
import math
from models.pinsage_pytorch.feature_copy import assign_features_to_blocks

from torch._C import import_ir_module
from models.pinsage_pytorch.lr_scheduler import Polynominal
from utils.helper import early_stopping
from utils.batch_test import test_one_prescrp
from models.pinsage_pytorch import layers
import torch
import torch.nn as nn 
from models.pinsage_pytorch import sampler as sampler_module
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import more_itertools
from time import time
import random 
import os
import tensorboardX as tb
import torchviz
from torchviz import make_dot, make_dot_from_trace


class PinSAGEModel(nn.Module):

    def __init__(self, full_graph, input_dim, hidden_dims, output_dims, num_layers, 
            entity_sampler, other_sampler, data_generator, device):
        super().__init__()

        self.g = full_graph
        self.data_generator = data_generator
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dims[num_layers-1]
        self.proj = layers.LinearProjector(full_graph, input_dim)
        self.sage = layers.MultiConvolveNet(input_dim, hidden_dims, output_dims, num_layers)
        self.si = layers.SyndromeInduction(output_dims[num_layers-1])
        self.entity_sampler = entity_sampler
        self.other_sampler = other_sampler


    def forward(self, entity_seeds, other_seeds, batch_entity_sets, batch_entity_adj):
        num_entities, num_others, device = self.data_generator.n_symptoms, self.data_generator.n_herbs, self.device
        entity_blocks = self.entity_sampler.sample_blocks(entity_seeds)
        other_blocks = self.other_sampler.sample_blocks(other_seeds)

        for i in range(len(entity_blocks)):
            entity_blocks[i] = entity_blocks[i].to(device)
            other_blocks[i] = other_blocks[i].to(device)
        self.g = self.g.to(device)
        assign_features_to_blocks(entity_blocks, self.g,  'sympt')
        assign_features_to_blocks(other_blocks, self.g, 'herb')

        self.batch_sympt_embeddings = self.get_representation(entity_blocks, ntype='sympt')
        self.batch_herb_embeddings = self.get_representation(other_blocks, ntype='herb')

        self.whole_sympt_embeddings = torch.from_numpy( np.zeros((num_entities, self.output_dim)).astype(np.float32) ).to(device)
        self.whole_sympt_embeddings[entity_seeds] = self.batch_sympt_embeddings

        self.whole_herb_embeddings = torch.from_numpy( np.zeros((num_others, self.output_dim)).astype(np.float32) ).to(device)
        self.whole_herb_embeddings[other_seeds] = self.batch_herb_embeddings
        
        self.batch_sympt_sum_embeddings = torch.matmul(batch_entity_sets, self.whole_sympt_embeddings)
        self.batch_sympt_mean_embeddings = torch.matmul(batch_entity_adj, self.batch_sympt_sum_embeddings)
        # self.batch_syndrome_embeddings = self.si(self.batch_sympt_mean_embeddings)
        self.batch_syndrome_embeddings = self.batch_sympt_mean_embeddings
        self.batch_predicted_other = torch.matmul(self.batch_syndrome_embeddings, self.whole_herb_embeddings.t())
        return self.batch_predicted_other

    def get_representation(self, blocks, ntype):
        h_item = self.proj(blocks[0].srcdata, ntype)
        # 获取最后一个block的终端节点特征
        h_item_dst = self.proj(blocks[-1].dstdata, ntype)
        # 如果是此种情况，则必须要求最初始的嵌入维度input_dim与output_dim是相同的
        # 比如input_dim=64，则 【【64,128】，【128,64】】是可行的
        # 而【【64,128】，【128,256】】则是不可行的
        return h_item_dst + self.sage(blocks, h_item)
        # 如果是这种情况，则不需要相同
        # return self.sage(blocks, h_item)

def get_weight_vector(data_generator):

    herb_freq_vector = list(np.zeros(dtype=np.float32, shape=(data_generator.n_herbs,)))
    for prescrp in data_generator.train_prescrp_list:
        herb_set = prescrp.herbs
        for herb in herb_set:
            herb_freq_vector[int(herb)] += 1.
    # 根据频次计算权重
    herb_freq_vector = np.asarray(herb_freq_vector)
    herb_weight_vector = (max(herb_freq_vector) + 1.0) /( herb_freq_vector + 1.0)
    herb_weight_vector = herb_weight_vector.reshape([1, data_generator.n_herbs]).astype(dtype=np.float32)
    # print('herb_weight_vector', herb_weight_vector)
    # print('herb_weight_vector type:', type(herb_freq_vector))
    # print('herb_weight_vector shape:', herb_freq_vector.shape)

    return herb_freq_vector


def cal_multi_label_loss(batch_predicted_other, batch_other_sets, data_generator, args, model, device):
    herb_freq_vector = torch.from_numpy(get_weight_vector(data_generator).astype(np.float32)).unsqueeze(1).to(device)
    embed_loss = torch.matmul(torch.square(torch.subtract(batch_predicted_other, batch_other_sets)), 
                               herb_freq_vector)
    embed_loss = torch.mean(embed_loss)

    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += torch.norm(param, p=2)
    reg_loss = reg_loss * eval(args.regs)[0]

    total_loss = reg_loss  + embed_loss
    return total_loss, embed_loss, reg_loss

def set_rand_seed(seed=1):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   # 保证每次返回得的卷积算法是确定的

def batch_train():
    pass

def batch_test():
    pass

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def count_params(model):
    model_parameters = filter(lambda p:p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('model total params is:', num_params)
    return num_params 


def pinsage_train(data_generator, args, g):
    device = torch.device(args.device)
     
    set_rand_seed(seed=10)

    prescrp_train_batch_sampler = sampler_module.PrescrpTrainBatchSampler(
        g, data_generator
    )
    prescrp_train_whole_sampler = sampler_module.PrescrpTrainWholeSampler(
        g, data_generator
    )
    prescrp_train_seq_sampler = sampler_module.PrescrpTrainSeqSampler(
        g, data_generator, args.batch_size
    )
    prescrp_test_batch_sampler = sampler_module.PrescrpTestBatchSampler(
        g, data_generator, args.batch_size
    )
    prescrp_test_whole_sampler = sampler_module.PrescrpTestWholeSampler(
        g, data_generator
    )

    # 构造症状节点的邻居采样器
    sympt_neighbor_sampler = sampler_module.NeighborSampler(
        g, 'sympt', 'herb',args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    # 构造草药节点的邻居采样器
    herb_neighbor_sampler = sampler_module.NeighborSampler(
        g, 'herb','sympt', args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = sampler_module.PinSAGECollator(sympt_neighbor_sampler, 
                                             herb_neighbor_sampler,
                                             g, 
                                             'sympt', 
                                             'herb', )

                                            
    # 在训练集上的处方采样器，采用乱序
    train_dataloader = DataLoader(
        prescrp_train_batch_sampler,
        batch_size=1,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    whole_train_dataloader = DataLoader(
        prescrp_train_whole_sampler, 
        batch_size=1,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers
    )
    train_seq_dataloader = DataLoader(
        prescrp_train_seq_sampler, 
        batch_size=1,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers
    )

    # 在测试集上的处方采样器，采用顺序
    test_dataloader = DataLoader(
        prescrp_test_batch_sampler,
        batch_size=1,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    whole_test_dataloader = DataLoader(
        prescrp_test_whole_sampler, 
        batch_size=1,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers
    )
    
    train_dataloader_it = iter(train_dataloader)
    test_dataloader_it = iter(test_dataloader)
    whole_train_dataloader_it = iter(whole_train_dataloader)
    train_seq_dataloader_it = iter(train_seq_dataloader)
    whole_test_dataloader_it = iter(whole_test_dataloader)

    # test_entity_seeds, test_other_seeds, test_entity_blocks, test_other_blocks, \
    #     test_batch_entity_sets, test_batch_other_sets, test_batch_entity_adj = next(test_dataloader_it)

    # Model
    model = PinSAGEModel(g, args.input_dim,  eval(args.hidden_dims), 
                        eval(args.output_dims), args.num_layers, 
                        sympt_neighbor_sampler, herb_neighbor_sampler, 
                        data_generator, device
                        ).to(device)
    count_params(model)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lambda_rule(epoch):
        power = 0.95
        return args.lr * 1 * (1 -  float(epoch) / args.epoch) ** power
    def lambda_rule_2(epoch):
        iter_times = epoch // 500
        return args.lr * 0.50 ** iter_times

    scheduler =  torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule_2)
    lr_scheduler = Polynominal(
        optimizer=opt,
        step_size=10,
        iter_max=args.epoch,
        power=0.9
    )

    perf_dir = '/home/exp/SMGCN/SMGCN/data/pinsage-pytorch/performances/'
    # args_info_str =  str(args.lr) + "-" + str(args.epoch) + "-" + str(args.regs)
    args_info_str =  str(args.lr) + "-" + str(args.epoch) + "-" + str(args.regs) + "|" +\
         str(args.sympt_threshold) + "-" + str(args.herb_threshold)  + "|" + \
             str(args.hidden_dims) + str(args.output_dims) + "|" + str(args.node_dropout) + "|" + str(args.mess_dropout)
    perf_dir = os.path.join(perf_dir, args_info_str)
    ensure_dir(perf_dir)
    train_perf_save_path = os.path.join(perf_dir, "train_perf.txt")
    test_perf_save_path = os.path.join(perf_dir, "test_perf.txt")
    with open(train_perf_save_path, 'w') as f:
        f.write("args settings is：\n  {0}\n".format(str(args)))
    with open(test_perf_save_path, 'w') as f:
        f.write("args settings is: \n  {0}\n".format(str(args)))

    embeddings_dir = '/home/exp/SMGCN/SMGCN/data/pinsage-pytorch/embeddings/'
    embeddings_dir = os.path.join(embeddings_dir, args_info_str)
    ensure_dir(embeddings_dir)
    sympt_embeddings_path = os.path.join(embeddings_dir, 'sympt_embeddings.npy')
    herb_embeddings_path = os.path.join(embeddings_dir, 'herb_embeddings.npy')
    # mlp_layer_weight_path = os.path.join(embeddings_dir, 'mlp_layer_weight.npy')
    # mlp_layer_bias_path = os.path.join(embeddings_dir, 'mlp_layer_bias.npy')

    checkpoints_dir = '/home/exp/SMGCN/SMGCN/data/pinsage-pytorch/checkpoints'
    checkpoints_dir = os.path.join(checkpoints_dir, args_info_str)
    ensure_dir(checkpoints_dir)
    checkpoints_save_path = os.path.join(checkpoints_dir, "best_perf_checkpoint.pth")

    logs_dir = '/home/exp/SMGCN/SMGCN/data/pinsage-pytorch/logs'
    logs_dir = os.path.join(logs_dir, args_info_str)
    ensure_dir(logs_dir)
    writer = tb.SummaryWriter(logs_dir, comment='pinsage-pytorch', flush_secs=25)
    # whole_test_entity_seeds, whole_test_other_seeds, whole_test_entity_blocks, whole_test_other_blocks,\
    #      whole_test_entity_sets, whole_test_other_sets, whole_test_entity_adj = next(whole_test_dataloader_it)
    # whole_test_entity_sets  =  torch.from_numpy(whole_test_entity_sets).to(device)
    # whole_test_other_sets = torch.from_numpy(whole_test_other_sets).to(device)
    # whole_test_entity_adj = torch.from_numpy(whole_test_entity_adj).to(device)
    # for i in range(len(whole_test_entity_blocks)):
    #     whole_test_entity_blocks[i] = whole_test_entity_blocks[i].to(device)
    #     whole_test_other_blocks[i] = whole_test_other_blocks[i].to(device)
    
        
    # for name,param in model.named_parameters():
    #     print('name: ', name)
    #     print('param:', param)
    #     print('\n')

    # print('getting modules')
    # for name, module in model.named_modules():
    #     print('name：', name, " module:", module)


    # 记录每10次迭代时在测试集上的准确度、回召率等平均值,训练集和测试集上的损失
    precision_logger, recall_logger, ndcg_logger, hit_ratio_logger = [], [], [], []
    test_loss_logger = []
    train_loss_logger = []
    # 设置初始的召回率最初值 和 初始的步骤数
    recall_best_value_pre = 0
    stopping_step = 0

    times = args.epoch

    entity_seeds, other_seeds,entity_blocks, other_blocks, \
            entity_sets, other_sets, entity_adj = next(whole_train_dataloader_it)
   
    entity_sets  =  torch.from_numpy(entity_sets).to(device)
    other_sets = torch.from_numpy(other_sets).to(device)
    entity_adj = torch.from_numpy(entity_adj).to(device)

    y = model(entity_seeds, other_seeds,
              entity_sets, entity_adj, 
            )
    # g = torchviz.make_dot(y, params=dict(model.named_parameters()))
    g = torchviz.make_dot(y)
    g.render('pinsage-pytorch', view=False)
    # with writer:
    #     writer.add_graph(model, 
    #             (entity_seeds, other_seeds,
    #              entity_sets, entity_adj, 
    #             ), verbose=False)
   
    t0 = time()
    for epoch_id in range(times):

        model.train()

        t1 = time()
        
        # 全部进行训练
        
        predicted_other = model(entity_seeds, other_seeds,
                                entity_sets, entity_adj)
        total_loss, embed_loss, reg_loss = cal_multi_label_loss(predicted_other, 
                                                        other_sets, data_generator, args, model, device)
        train_performance_str = '[Train Epoch=%d ] train_total_loss=%.4f, train_embed_loss=%.4f, train_reg_loss=%.4f' % (
            epoch_id,  total_loss, embed_loss, reg_loss)
        print(train_performance_str)
        with open(train_perf_save_path, 'a+') as f:
            f.write("{0}\n\n ".format(train_performance_str))
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        scheduler.step()
        # lr_scheduler.step()

        writer.add_scalars('loss/train', {'total_loss':total_loss , 
                                          'embed_loss': embed_loss, 
                                          'reg_loss': reg_loss
                                         }, epoch_id)

        # 保留本次模型中的训练参数
        whole_sympt_embeddings = model.whole_sympt_embeddings.data.cpu().numpy()
        whole_herb_embeddings = model.whole_herb_embeddings.data.cpu().numpy()
        mlp_layer_weight = model.si.mlp.weight.data.cpu().numpy()
        mlp_layer_bias = model.si.mlp.bias.data.cpu().numpy()

        # # 批量方式进行训练1 从训练集随机进行的
        # train_total_loss = 0.0
        # train_embed_loss = 0.0
        # train_reg_loss = 0.0
        # for batch_id in tqdm.trange(args.batches_per_epoch):
        #     entity_seeds, other_seeds,entity_blocks, other_blocks, batch_entity_sets, batch_other_sets, batch_entity_adj = next(train_dataloader_it)

        #     # Copy to GPU
        #     for i in range(len(entity_blocks)):
        #         entity_blocks[i] = entity_blocks[i].to(device)
        #         other_blocks[i] = other_blocks[i].to(device)

        #     batch_entity_sets  =  torch.from_numpy(batch_entity_sets).to(device)
        #     batch_other_sets = torch.from_numpy(batch_other_sets).to(device)
        #     batch_entity_adj = torch.from_numpy(batch_entity_adj).to(device)
        #     batch_predicted_other = model(entity_seeds, other_seeds,
        #                                   
        #                                  batch_entity_sets, batch_entity_adj, 
        #                                  data_generator.n_symptoms, data_generator.n_herbs, device)
        #     batch_total_loss, batch_embed_loss, batch_reg_loss = cal_multi_label_loss(batch_predicted_other, 
        #                                                     batch_other_sets, data_generator, args, model, device)

        #     train_total_loss += batch_total_loss
        #     train_embed_loss += batch_embed_loss
        #     train_reg_loss += batch_reg_loss
        #     print('[Train Epoch=%d batch=%d] train_total_loss=%.4f, train_embed_loss=%.4f, train_reg_loss=%.4f' % (
        #         epoch_id, batch_id, batch_total_loss, batch_embed_loss, batch_reg_loss))
        #     opt.zero_grad()
        #     batch_total_loss.backward()
        #     opt.step()

        # print("-"*100)
        # train_performance_str = 'Train Epoch %d [time=%.1fs]: total_loss=%.5f, embed_loss=%.5f, reg_loss=%.5f\n ' \
        #                                 % ( epoch_id, time()-t1, train_total_loss, train_embed_loss, train_reg_loss)
        # print(train_performance_str)
        # with open(train_perf_save_path, 'a+') as f:
        #     f.write("{0}\n\n ".format(train_performance_str))
        # print("-"*100)

        # # 批量方式进行训练2 即每次批量处理是从训练集顺序进行的
        # train_total_loss = 0.0
        # train_embed_loss = 0.0
        # train_reg_loss = 0.0
        # train_batch_num = math.ceil(data_generator.n_train / args.batch_size)
        # for batch_id in tqdm.trange(train_batch_num):
        #     entity_seeds, other_seeds,entity_blocks, other_blocks, batch_entity_sets, batch_other_sets, batch_entity_adj = next(train_seq_dataloader_it)

        #     # Copy to GPU
        #     for i in range(len(entity_blocks)):
        #         entity_blocks[i] = entity_blocks[i].to(device)
        #         other_blocks[i] = other_blocks[i].to(device)

        #     batch_entity_sets  =  torch.from_numpy(batch_entity_sets).to(device)
        #     batch_other_sets = torch.from_numpy(batch_other_sets).to(device)
        #     batch_entity_adj = torch.from_numpy(batch_entity_adj).to(device)
        #     batch_predicted_other = model(entity_seeds, other_seeds,
        #                                  
        #                                  batch_entity_sets, batch_entity_adj, 
        #                                  )
        #     batch_total_loss, batch_embed_loss, batch_reg_loss = cal_multi_label_loss(batch_predicted_other, 
        #                                                     batch_other_sets, data_generator, args, model, device)

        #     train_total_loss += batch_total_loss
        #     train_embed_loss += batch_embed_loss
        #     train_reg_loss += batch_reg_loss
        #     print('[Train Epoch=%d batch=%d] train_total_loss=%.4f, train_embed_loss=%.4f, train_reg_loss=%.4f' % (
        #         epoch_id, batch_id, batch_total_loss, batch_embed_loss, batch_reg_loss))
        #     opt.zero_grad()
        #     batch_total_loss.backward()
        #     opt.step()
        #     scheduler.step()

        # print("-"*100)
        # train_performance_str = 'Train Epoch %d [time=%.1fs]: total_loss=%.5f, embed_loss=%.5f, reg_loss=%.5f\n ' \
        #                                 % ( epoch_id, time()-t1, train_total_loss, train_embed_loss, train_reg_loss)
        # print(train_performance_str)
        # with open(train_perf_save_path, 'a+') as f:
        #     f.write("{0}\n\n ".format(train_performance_str))
        # print("-"*100)


        # Evaluate
        t2 = time()
        model.eval()
        test_total_loss = 0.0
        test_embed_loss = 0.0
        test_reg_loss = 0.0


        if (epoch_id+1) % 10 == 0:
            with torch.no_grad():
                # whole_test_predicted_other = model(whole_test_entity_seeds, whole_test_other_seeds,
                #                                  
                #                                  whole_test_entity_sets, whole_test_entity_adj, 
                #                                  
                #                             )

                # 需要计算在测试集上的测试损失以及相应的评价指标
                # 使用批处理的方法
                # num_test_batches = more_itertools.ilen(test_dataloader_it)
                num_test_batches = math.ceil(data_generator.n_test / args.batch_size)
                print("num_test_batches is", num_test_batches)

                len_Ks = len(eval(args.Ks))
                test_result = {'precision': np.zeros(len_Ks), 'recall': np.zeros(len_Ks),
                    'ndcg': np.zeros(len_Ks), 'hit_ratio': np.zeros(len_Ks)}

                
                # dataloader_len = more_itertools.ilen(test_dataloader_it)
                # print("The len of dataloader is:", dataloader_len
                for i_test_batch in range(num_test_batches):
                    try:
                        test_entity_seeds, test_other_seeds, test_entity_blocks, test_other_blocks, \
                            test_batch_entity_sets, test_batch_other_sets, test_batch_entity_adj = next(test_dataloader_it)
                    except StopIteration:
                        pass

                    
                    # print('current i_batch is:', i_test_batch)
                   
                    test_batch_entity_sets  =  torch.from_numpy(test_batch_entity_sets).to(device)
                   
                    test_batch_other_sets = torch.from_numpy(test_batch_other_sets).to(device)
                    
                    test_batch_entity_adj = torch.from_numpy(test_batch_entity_adj).to(device)

                    test_batch_predicted_other = model(test_entity_seeds, test_other_seeds,
                                            # test_entity_blocks, test_other_blocks, 
                                            test_batch_entity_sets, test_batch_entity_adj, 
                                            )
                    batch_total_loss, batch_embed_loss, batch_reg_loss = cal_multi_label_loss(test_batch_predicted_other, 
                                                                test_batch_other_sets, data_generator, args, model, device)
                    test_total_loss += batch_total_loss
                    test_embed_loss += batch_embed_loss
                    test_reg_loss += batch_reg_loss

                    print('[Test Epoch=%d batch=%d] batch_total_loss=%.4f, batch_embed_loss=%.4f, batch_reg_loss=%.4f' % (
                        epoch_id, i_test_batch, batch_total_loss, batch_embed_loss, batch_reg_loss) )

                    predict_herbs_vs_true_herbs = zip(test_batch_predicted_other, test_batch_other_sets)
                
                    batch_test_res = []
                    for x in predict_herbs_vs_true_herbs:
                        x_res = test_one_prescrp(x, data_generator, args)
                        batch_test_res.append(x_res)
                    
                    for res in batch_test_res:
                        test_result['precision'] += res['precision']
                        test_result['recall'] += res['recall']
                        test_result['ndcg'] += res['ndcg']
                        test_result['hit_ratio'] += res['hit_ratio']

                test_result['precision'] /= data_generator.n_test
                test_result['recall'] /= data_generator.n_test
                test_result['ndcg'] /= data_generator.n_test
                test_result['hit_ratio'] /= data_generator.n_test

                writer.add_scalars('test/scores_k=5', {
                    'precision': test_result['precision'][0],
                    'recall': test_result['recall'][0],
                    'ndcg': test_result['ndcg'][0],
                    'hit_ratio': test_result['hit_ratio'][0],
                }, epoch_id)

                writer.add_scalars('test/scores_k=10', {
                    'precision': test_result['precision'][1],
                    'recall': test_result['recall'][1],
                    'ndcg': test_result['ndcg'][1],
                    'hit_ratio': test_result['hit_ratio'][1],
                }, epoch_id)

                writer.add_scalars('test/scores_k=20', {
                    'precision': test_result['precision'][2],
                    'recall': test_result['recall'][2],
                    'ndcg': test_result['ndcg'][2],
                    'hit_ratio': test_result['hit_ratio'][2],
                }, epoch_id)
                
                test_performance_str = 'Test Epoch %d [time=%.1fs]:\n ' \
                                    'test_average_loss=%.5f, ' \
                                    'precision=[%s], recall=[%s], ' \
                                    'ndcg=[%s]' % \
                                    (epoch_id , time()-t2, float(test_total_loss / num_test_batches),                                    
                                    '\t'.join(['%.5f' % r for r in test_result['precision']]),
                                        '\t'.join(['%.5f' % r for r in test_result['recall']]),
                                        '\t'.join(['%.5f' % r for r in test_result['ndcg']]),
                                        )
                print(test_performance_str)  
                with open(test_perf_save_path, 'a+') as f:
                    f.write("{0}\n\n ".format(test_performance_str))
                
                writer.add_scalars('loss/test', {'total_loss': float(test_total_loss / num_test_batches),
                                                 'embed_loss': float(test_embed_loss / num_test_batches),
                                                 'reg_loss': float(test_reg_loss / num_test_batches)
                                                 }, epoch_id)

                recall_best_value_pre, stopping_step, should_stop = early_stopping(test_result['recall'][0],
                                                                            recall_best_value_pre, stopping_step)
                if should_stop == True:
                    break

                # 如果结果比上次好，则保存模型检查点， 还有embeddings和mlp的参数
                if stopping_step == 0:
                    torch.save(model.state_dict(), checkpoints_save_path)
                    print('**************Checkpoints Saved !*************')
                    np.save(sympt_embeddings_path, whole_sympt_embeddings)
                    np.save(herb_embeddings_path, whole_herb_embeddings)
                    # np.save(mlp_layer_weight_path, mlp_layer_weight)
                    # np.save(mlp_layer_bias_path, mlp_layer_bias)
                    print('***************Embeddings Saved !**************')

                train_loss_logger.append(total_loss)
                test_loss_logger.append(test_total_loss)
                precision_logger.append(test_result['precision'])
                recall_logger.append(test_result['recall'])
                ndcg_logger.append(test_result['ndcg'])
                hit_ratio_logger.append(test_result['hit_ratio'])

          
    #将列表转化为数组
    pres = np.array(precision_logger)
    recalls = np.array(recall_logger)
    ndcgs = np.array(ndcg_logger)
    train_losses = np.array(train_loss_logger)
    test_losses = np.array(test_loss_logger)

    # 找出测试结果中召回率最好的那一次迭代
    max_recall = max(recalls[:,0])
    max_rcl_idx = list(recalls[:,0]).index(max_recall)
    # 打印相关信息
    final_performance_str = 'Best Iter (Epoch %d):\n total_time: %.1fs \n recalls=[%s], precisions=[%s],' \
                            'ndcgs=[%s]' % (max_rcl_idx, time()-t0,
                                            '\t'.join(['%.5f' % r for r in recalls[max_rcl_idx]]),
                                            '\t'.join(['%.5f' % r for r in pres[max_rcl_idx]]),
                                            '\t'.join(['%.5f' % r for r in ndcgs[max_rcl_idx]]))
    print(final_performance_str)
    with open(test_perf_save_path, 'a+') as f:
        f.write("{0}\n\n ".format(final_performance_str))
