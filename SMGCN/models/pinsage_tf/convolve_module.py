import tensorflow as tf
import dgl
import dgl.function as fn
import numpy as np

class ConvolveNet(tf.keras.Model):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0, act=tf.keras.layers.LeakyReLU()):
        super(ConvolveNet, self).__init__()
        self.Q = tf.keras.layers.Dense(units = hidden_dim, 
                                       activation = act,
                                       use_bias = False,
                                       kernel_initializer=tf.initializers.GlorotUniform(),
                                       )
        self.W = tf.keras.layers.Dense(units = output_dim, 
                                       activation = act, 
                                       use_bias = False,
                                       kernel_initializer=tf.initializers.GlorotUniform(),
                                       )
        tf.random.set_seed(0)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)


    def call(self, g, h, weights):
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.Q(self.dropout(h_src))
            g.edata['w'] = tf.cast(weights, tf.float32)
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            # ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            ws = tf.clip_by_value( tf.expand_dims(g.dstdata['ws'], 1), 1, np.inf)
            z = self.W(self.dropout( tf.concat([n / ws, h_dst], 1) ) )
            # z_norm = tf.Variable(z.norm(2, 1, keepdim=True))
            z_norm = tf.Variable(tf.norm(z, ord=2, axis=1, keepdims=True))
            for index in range(tf.shape(z_norm)[0]):
                tf.compat.v1.assign(z_norm[index], tf.convert_to_tensor([1.]))
            z = z / z_norm
            return z


class MultiConvolveNet(tf.keras.Model):
    def __init__(self, hidden_dim, n_layers):
        super(MultiConvolveNet, self).__init__()

        self.convs = []
        for _ in range(n_layers):
            self.convs.append(ConvolveNet(hidden_dim, hidden_dim, hidden_dim))

    def call(self, blocks, h):
        # h对应初始所有采样的样本节点
        # 也就是block[0]的srcdata
        for layer, block in zip(self.convs, blocks):
            # h_dst是要需要更新表示的那些节点
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h

        
        

    



