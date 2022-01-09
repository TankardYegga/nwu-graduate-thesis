import torch
import dgl
import dgl.function as fn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConvolveNet(nn.Module):


    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dim, hidden_dim)
        self.W = nn.Linear(input_dim + hidden_dim, output_dim)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout_rate)


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)
    

    def forward(self, g, h, weights):
        """
        g : graph 
        h : node features  
        weights : scalar edge weights  
        """
        h_src, h_dst = h 
        # print('h_src shape:', h_src.shape)
        # print('h_dst shape:', h_dst.shape)
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()     
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']  
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            # print('h_src n shape:', n.shape)
            # print('h_dst shape:', h_dst.shape)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            # 将z_norm中的0换为1
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z



class MultiConvolveNet(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dims, num_layers):
        super(MultiConvolveNet, self).__init__()

        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            self.convs.append(ConvolveNet(input_dim, 
                                         hidden_dims[layer_idx], 
                                         output_dims[layer_idx])
                                         )
            input_dim = output_dims[layer_idx]


    def forward(self, blocks, h):
        # h对应初始所有采样的样本节点
        # 也就是block[0]的srcdata
        for layer, block in zip(self.convs, blocks):
            # h_dst是要需要更新表示的那些节点
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h

        
        """
        

        Returns:
            [type]: [description]
        """
# 为图节点的每个属性创建对应的线性映射器
def _init_input_modules(g, ntype, input_dim):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()

    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], input_dim)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(
                data.max() + 1, input_dim, padding_idx=-1)
            # 注意embedding函数实际索引时只能使用0~【data.max() + 2 - 1 = data.max() + 1】
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            module_dict[column] = m

    return module_dict
    
class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """
    def __init__(self, full_graph, input_dim):
        super().__init__()

        self.sympt_inputs = _init_input_modules(full_graph, 'sympt', input_dim)
        self.herb_inputs = _init_input_modules(full_graph, 'herb', input_dim)

    def forward(self, ndata, ntype):
        projections = []
        self.inputs = self.sympt_inputs if ntype == 'sympt' else self.herb_inputs

        for feature, data in ndata.items():
            # print('The feature is:', feature)
            # print('The data shape is:', data.shape)
            if feature == dgl.NID:
                continue

            module = self.inputs[feature]
            result = module(data)
            projections.append(result)
        
        # 节点的每个特征表示经过线性映射都恢复到了同一纬度
        # 然后把这些特征表示直接进行按元素相加
        # 就得到每个节点的最终特征表示
        return torch.stack(projections, 1).sum(1)


class SyndromeInduction(nn.Module):

    def __init__(self, output_dim, act=F.relu):
        super().__init__()
        self.mlp = nn.Linear(output_dim, output_dim)
        self.act = act

    def forward(self, batch_sympt_mean_embeddings):
        batch_syndrome_embeddings = self.act(self.mlp(batch_sympt_mean_embeddings))
        return batch_syndrome_embeddings


