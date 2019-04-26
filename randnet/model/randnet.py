import tensorflow as tf
from randnet.model.layers import Node, RandomWiring

from tensorflow.python import keras


class RandNet(keras.Model):
    def __init__(self, num_classes):
        super(RandNet, self).__init__(name='rand_net')
        self.randwire_1 = RandomWiring(channels=16, random_graph_algorithm="ws")
        self.randwire_2 = RandomWiring(channels=32, random_graph_algorithm="ws")
        self.global_average_pool = keras.layers.GlobalAveragePooling2D()
        self.classify = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.randwire_1(inputs)
        x = self.randwire_2(x)
        x = self.global_average_pool(x)
        x = self.classify(x)
        return x
