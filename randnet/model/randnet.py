import tensorflow as tf
from randnet.model.layers import Node, RandomWiring

from tensorflow.python import keras


class RandNet(keras.Model):
    def __init__(self, num_classes):
        super(RandNet, self).__init__(name='rand_net')
        self.randwire_1 = RandomWiring(16, 2, "ws", n=32, p=0.75, k=4, m=0)

    def call(self, inputs, training=None, mask=None):
        x0 = self.node_op_0([inputs])
        x1_1 = self.node_op_1_1([x0])
        x1_2 = self.node_op_1_2([x0])
        x1_3 = self.node_op_1_3([x0])
        x2 = self.node_op_2([x1_1, x1_2, x1_3])
        out = self.global_average_pool(x2)
        out = self.classify(out)
        return out
