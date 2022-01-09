import tensorflow as tf
class GraphSageBase(tf.keras.Model):

    def __init__(self, raw_features, internal_dim, num_layers, last_has_activ):

        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()

        self.input_layer = RawFeature(raw_features, name="raw_feature_layer")

        self.seq_layers = []
        for i in range (1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = internal_dim if i > 1 else raw_features.shape[-1]
            has_activ = last_has_activ if i == num_layers else True
            aggregator_layer = MeanAggregator ( input_dim
                                              , internal_dim
                                              , name=layer_name
                                              , activ = has_activ
                                              )
            self.seq_layers.append(aggregator_layer)

    def call(self, minibatch):
        # 取出当前batch中用到的所有节点
        x = self.input_layer(tf.squeeze(minibatch.src_nodes))
        for aggregator_layer in self.seq_layers:
            # 逐层聚合
            x = aggregator_layer ( x
                                 , minibatch.dstsrc2srcs.pop()
                                 , minibatch.dstsrc2dsts.pop()
                                 , minibatch.dif_mats.pop()
                                 )
        return x # shape: (batch_size, src_dim)

class GraphSageSupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, num_classes):
        super().__init__(raw_features, internal_dim, num_layers, True)
        self.classifier = tf.keras.layers.Dense ( num_classes
                                                , activation = tf.nn.softmax
                                                , use_bias = False
                                                , kernel_initializer = init_fn
                                                , name = "classifier"
                                                )

    def call(self, minibatch):
        return self.classifier( super().call(minibatch) )
