import dgl
import dgl.function as fn
import tensorflow as tf 
import tensorflow.compat.v1 as tf1


def compact_and_copy(frontier, entity_seeds):
    block = dgl.to_block(frontier, dst_nodes=entity_seeds)

    # for col,data in frontier.edata.items():
    #     if col == dgl.NID:
    #         continue
    #     print(type(block.edata[dgl.EID]))
    #     print(block.edata[dgl.EID])
    #     print(block.edata[dgl.EID].shape)
        # print(data[tf1.convert_to_tensor(3)])
        # print(data[tf.cast( tf.convert_to_tensor([3]), tf.int64 ) ])
        # print(data[tf1.cast(  tf1.convert_to_tensor([3]) , tf1.int64 )])

        #block.edata[col] = data[tf1.convert_to_tensor(3)]
        
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        # tf中的tensor不能直接索引
        idx = tf.expand_dims(block.edata[dgl.EID], 1)
        idx = tf.cast(idx, tf.int64)
        block.edata[col] = tf.gather_nd(data, idx)
    return block


def assign_simple_node_features(ndata, g, ntype, assign_ids=False):
    induced_nodes = ndata[dgl.NID]
    induced_nodes = tf.cast(induced_nodes, tf.int64)
    induced_nodes = tf.expand_dims(induced_nodes, 1)

    for col in g.nodes[ntype].data.keys():
        if not assign_ids and col == dgl.NID:
            continue
        ndata[col] = tf.gather_nd(g.nodes[ntype].data[col], induced_nodes)
        # ndata[col] = g.nodes[ntype].data[col][induced_nodes]


class NeighborSampler(object):
    def __init__(self, g, entity_type, other_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.entity_type = entity_type
        self.other_type = other_type
        self.entity_to_other_type = list(g.metagraph()[entity_type][other_type])[0]
        self.other_to_entity_type = list(g.metagraph()[other_type][entity_type])[0]

        self.samplers = [
            dgl.sampling.PinSAGESampler(g, entity_type, other_type, random_walk_length, random_walk_restart_prob,
                                num_random_walks, num_neighbors, weight_column="weights")
        ]


    def sample_blocks(self, entity_seeds):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(entity_seeds)
            block = compact_and_copy(frontier, entity_seeds)
            entity_seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks


    @classmethod
    def assign_features_to_blocks(cls, blocks, g, entity_type, assign_ids=False):
        assign_simple_node_features(blocks[0].srcdata, g, entity_type, assign_ids)
        assign_simple_node_features(blocks[-1].dstdata, g, entity_type, assign_ids)


    

        