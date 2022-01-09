from numpy.core.fromnumeric import trace
from .feature_copy import assign_features_to_blocks
import dgl
import torch
from torch.utils.data import IterableDataset
import math
import traceback
import sys


"""
用最简单的话说 block结构就是一组初始节点和它们利用PinSage采样器获得的邻居结点
组成的二部图连接结构

这里的二部图是同质图进行改造的
与异质图的二部图结构是有区别的
异质图的二部图结构当中只有两边不同的结点之间是有联系的，同种节点之间没有任何联系
但是在这里的情况是
一群初始种子节点它们通过PinSage采样器获取到的邻居节点就在它们这群种子节点当中
所以对于这些在初始种子节点内部的邻居关系在同质图-二部图结构上的表现就是
在二部图的右边也就是终端节点是有边相连的
"""
def compact_and_copy(frontier, seeds):
    # 映射为二部图结构
    # 疑问，所有的异质图都可以映射为二部图的结构吗？映射过程中会有边或者节点的损失吗？
    #  这样映射的目的又是什么呢？这样映射的二部图结构块是唯一的吗？
    #  为什么初始节点处要包含所有的终端节点呢、如果这样那么起始处的这些终端节点会有相关的边吗？
    # 注：在api文档说明中指出二部图结构块中终端节点必须有1条入边，注意入边是可以为任意类型的；
    #     起始的节点包含终端节点，和那些至少有1条出边的节点，注意出边必须连接的是终端节点；
    # 理解：1落单的节点肯定会被丢弃，没有出度但是有入度的节点必然会出现在终端节点那里，
    #       没有入度但是有出度的节点必然会出现在起始节点那里；
    #  2 映射的结果肯定是不唯一的，但是可以指定终端节点的id，以及是否将终端节点包含在起始节点处，
    #    这样得到的二部图结构可能就是唯一的了；但是终端节点必须都有入边，不然将会报错
    #   3 可以使用create_block方法获得创建二部图结构块更灵活的方法
    #   4 使用block会在过滤掉那些落单的节点后进行编号的重排  边的编号也可能会被重排
    #   5 默认不把终端节点加入到起始节点处
    block = dgl.to_block(frontier, seeds)
   
    # 经过实验证明得到的block其ndata[属性名]与srcdata[属性名]是一致的，与dstdata[属性名]不一致
    # 如果边的数量没有变化，block其edata也是一样
    # 疑问：这里为什么节点的特征不需要重新赋值？
    # 【解释与注意事项】：1.如果节点或者边的数量发生改变了就需要重新赋值；
    #            也可能数量不变，但是边的编号也可能会发生改变，所以也需要重新赋值 
    #  2. 需要注意的是否需要对节点或者边进行特征赋值，取决于to_block()的图参数g
    #    这里的frontier是PinSage对seeds种子节点采样后的结果，
    #    PinSAGESampler采样函数返回的异质图是不会包含节点的任何属性的，只会包含边上的权重
    #    所以这里得到的block自然也不会包含原图上节点的所有属性，但是边的权重保留
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block


class NeighborSampler(object):
    def __init__(self, g, entity_type, other_type,  random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.entity_type = entity_type
        self.other_type = other_type
        self.entity_to_other_etype = list(g.metagraph()[entity_type][other_type])[0]
        self.other_to_entity_etype = list(g.metagraph()[other_type][entity_type])[0]

        # 为每层都提供了一个PinSage采样器，而不是每层都使用同一个采样器
        # 但是这里每层采样器的参数都是一样的

        # PinSage采样器相比于random-walk采样器，最重要的参数是num_neignbors，
        # 即根据重要性来选择一定数量的邻居

        # 这里采样返回的是item节点的同质图，这个同质图的节点数量与原图g中的item节点数是一样的
        # 但是其含有的边就只连接相应节点群和它们的邻居节点
        # 边对应的起始节点为节点群的邻居节点，而终端节点为节点群
        # 返回的图的边中含有weights这个属性，代表访问数
        """注：PinSAGESampler是对双向的二部结构图进行采样 源节点的同质邻居
           也就是说可以通过两跳、4跳、6跳等来获取同质邻居
        """
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, entity_type, other_type, random_walk_length,
                random_walk_restart_prob, num_random_walks, num_neighbors, weight_column="weights")
            for _ in range(num_layers)]

    def sample_blocks(self, seeds):
        blocks = []
        # 这里的seeds就是节点群
        for sampler in self.samplers:
            # 得到该层该节点群采样后的同质图
            try:
                frontier = sampler(seeds)
            except Exception as e:
                sys.exc_info()
                traceback.print_exc()
                print(e)
                print('\n\n')
                print(Exception)

            # 对于采样异质图进行二部图结构化调整
            block = compact_and_copy(frontier, seeds)

            # 上一步的种子是block结构的dstnodes，这里使用block的srcdata[dgl.NID]来作为新的seeds
            # 需要注意：block的srcdata包含所有节点，及二部图结构图中起始和终端的节点都会包含在内
            #          而不是仅仅是起始的那些节点，
            #          这样一样seeds便会不断的扩大，不断地把每一次采样的邻居加入其内
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks


class PinSAGECollator(object):

    def __init__(self, entity_sampler, other_sampler, g, entity_type, other_type):
        self.entity_sampler = entity_sampler
        self.other_sampler = other_sampler
        self.entity_type = entity_type
        self.other_type = other_type
        self.g = g

    def collate_train(self, batches):
        # 这里的batches是list数据类型
        # print(type(batches))
        # print('batches is', batches)
        # print(batches[0])
       
        entity_seeds, other_seeds, batch_entity_sets, batch_other_sets, batch_entity_adj = batches[0]

        entity_seeds = torch.LongTensor( [int(entity_seed) for entity_seed in list(entity_seeds)] )
        other_seeds = torch.LongTensor( [int(other_seed) for other_seed in list(other_seeds)] )

        # Construct multilayer neighborhood via PinSAGE...
        entity_blocks = self.entity_sampler.sample_blocks(entity_seeds)
        other_blocks = self.other_sampler.sample_blocks(other_seeds)

        assign_features_to_blocks(entity_blocks, self.g,  self.entity_type)
        assign_features_to_blocks(other_blocks, self.g, self.other_type)

        return entity_seeds, other_seeds, entity_blocks, other_blocks, batch_entity_sets, batch_other_sets, batch_entity_adj


    def collate_test(self, batches):
        entity_seeds, other_seeds, batch_entity_sets, batch_other_sets, batch_entity_adj = batches[0]

        entity_seeds = torch.LongTensor( [int(entity_seed) for entity_seed in list(entity_seeds)] )
        other_seeds = torch.LongTensor( [int(other_seed) for other_seed in list(other_seeds)] )

        # Construct multilayer neighborhood via PinSAGE...
        entity_blocks = self.entity_sampler.sample_blocks(entity_seeds)
        other_blocks = self.other_sampler.sample_blocks(other_seeds)

        assign_features_to_blocks(entity_blocks, self.g,  self.entity_type)
        assign_features_to_blocks(other_blocks, self.g, self.other_type)

        return entity_seeds, other_seeds, entity_blocks, other_blocks, batch_entity_sets, batch_other_sets, batch_entity_adj


class PrescrpTrainBatchSampler(IterableDataset):

    def __init__(self, g, data_generator):
        self.g = g
        self.data_generator = data_generator

    def __iter__(self):
        while True:
            yield self.data_generator.sample_for_train(need_unique_sets=True)


class PrescrpTestBatchSampler(IterableDataset):

    def __init__(self, g, data_generator, batch_size):
        self.g = g
        self.batch_size = batch_size
        self.data_generator = data_generator

    def __iter__(self):
        num_test_batches = math.ceil(self.data_generator.n_test / self.batch_size)
        count = 0
        while True:
            yield self.data_generator.sample_for_test(need_unique_sets=True, i_batch=count)
            count = (count + 1) % num_test_batches


class PrescrpTrainSeqSampler(IterableDataset):

    def __init__(self, g, data_generator, batch_size):
        self.g = g
        self.batch_size = batch_size
        self.data_generator = data_generator

    def __iter__(self):
        count = 0
        num_train_batches = math.ceil(self.data_generator.n_train / self.batch_size)
        while True:
            yield self.data_generator.sample_for_sequential_train(need_unique_sets=True, i_batch=count)
            count = (count + 1) % num_train_batches


class PrescrpTestWholeSampler(IterableDataset):

    def __init__(self, g, data_generator):
        self.g = g
        self.data_generator = data_generator

    def __iter__(self):
        while True:
            yield self.data_generator.sample_for_whole_test(need_unique_sets=True)


class PrescrpTrainWholeSampler(IterableDataset):

    def __init__(self, g, data_generator):
        self.g = g
        self.data_generator = data_generator

    def __iter__(self):
        while True:
            yield self.data_generator.sample_for_whole_train(need_unique_sets=True)


