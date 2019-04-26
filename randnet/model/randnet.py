import tensorflow as tf
from randnet.model.layers import Node, RandomWiring

from tensorflow.python import keras


class RandNet(keras.Model):
    def __init__(self, num_classes):
        super(RandNet, self).__init__(name='rand_net')
        self.randwire_1 = RandomWiring(16, 2, "ws", n=32, p=0.75, k=4)
        self.global_average_pool = keras.layers.GlobalAveragePooling2D()
        self.classify = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.randwire_1(inputs)
        x = self.global_average_pool(x)
        out = self.classify(x)
        return out
