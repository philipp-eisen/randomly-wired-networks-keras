import tensorflow as tf
from randnet.model.layers import Node, RandomWiring

from tensorflow.python import keras


# N = 32
# C = 78 for small
# C = 109 or 154 in regular regime

class RandNetSmall(keras.Model):

    def __init__(self,
                 num_classes,
                 dropout_rate=0.2,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 seeds=(0, 1, 2),
                 channels=78,
                 n_nodes=32,
                 random_graph_algorithm="ws",
                 k=4,
                 p=0.75,
                 m=5
                 ):
        """

        Args:
            num_classes (int): The number of classes the dataset has
            bias_regularizer (keras.regularizers.Regularizer):  Regularizer function applied to all bias weights
            seeds (tuple(int)): the seeds used to seed the random graph generation algorithm of each of the three
                                RandomWiring layers.
            kernel_regularizer (keras.regularizers.Regularizer): Regularizer function applied to all kernel weights
            channels (int): the number of channels used in the first RandomWiring layer. The subsequent layer are
                            are going to be 2 * `channels` and 4 * `channels` respectively.
            random_graph_algorithm (str): The algorithm used to create the random graph for this `RandomWiring` stage.
                Can be one of the following:
                - `"ws"`: The Watts-Strogatz algorithm. If this algorithm is chosen the following
                    additional parameters have to be provided:
                    - `k`: Each node is joined with its `k` nearest neighbors in a ring
                    - `p`: The probability of rewiring each edge
                - `"ba"` Barabási–Albert algorithm. If this algorithm is chosen the following
                    additional parameters have to be provided:
                    - `m`: Number of edges to attach from a new node to existing nodes
                - `"er"`: Erdős-Rényi algorithm. If this algorithm is chosen the following
                    additional parameters have to be provided:
                    - `p`: probability of edge creation
            k (int): refer to "`random_graph_algorithm`" for explanation of the argument. Defaults to
                best performing value in arxiv 1904.01569.
            p (float): refer to "`random_graph_algorithm`" for explanation of the argument. Defaults to
                best performing value in arxiv 1904.01569.
            m (int): refer to "`random_graph_algorithm`" for explanation of the argument. Defaults to
                best performing value in arxiv 1904.01569.
        """

        super(RandNetSmall, self).__init__(name='rand_net')
        self.conv_1 = keras.layers.SeparableConv2D(filters=int(channels / 2),
                                                   kernel_size=(3, 3),
                                                   bias_regularizer=bias_regularizer,
                                                   kernel_regularizer=kernel_regularizer)
        self.batch_norm_1 = keras.layers.BatchNormalization()

        self.relu = keras.layers.ReLU()

        self.conv_2 = keras.layers.SeparableConv2D(filters=channels,
                                                   kernel_size=(3, 3),
                                                   bias_regularizer=bias_regularizer,
                                                   kernel_regularizer=kernel_regularizer)
        self.batch_norm_2 = keras.layers.BatchNormalization()

        self.randwire_1 = RandomWiring(channels=channels,
                                       n=n_nodes,
                                       bias_regularizer=bias_regularizer,
                                       kernel_regularizer=kernel_regularizer,
                                       seed=seeds[0],
                                       random_graph_algorithm=random_graph_algorithm,
                                       k=k,
                                       p=p,
                                       m=m)

        self.randwire_2 = RandomWiring(channels=2 * channels,
                                       n=n_nodes,
                                       bias_regularizer=bias_regularizer,
                                       kernel_regularizer=kernel_regularizer,
                                       seed=seeds[1],
                                       random_graph_algorithm=random_graph_algorithm,
                                       k=k,
                                       p=p,
                                       m=m)

        self.randwire_3 = RandomWiring(channels=4 * channels,
                                       n=n_nodes,
                                       bias_regularizer=bias_regularizer,
                                       kernel_regularizer=kernel_regularizer,
                                       seed=seeds[2],
                                       random_graph_algorithm=random_graph_algorithm,
                                       k=k,
                                       p=p,
                                       m=m)

        self.conv_out = keras.layers.SeparableConv2D(filters=1280,
                                                     kernel_size=(1, 1),
                                                     bias_regularizer=bias_regularizer,
                                                     kernel_regularizer=kernel_regularizer)
        self.batch_norm_out = keras.layers.BatchNormalization()

        self.global_average_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes,
                                     kernel_regularizer=bias_regularizer,
                                     bias_regularizer=kernel_regularizer)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.softmax = keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.batch_norm_1(x)

        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)

        x = self.randwire_1(x)
        x = self.randwire_2(x)
        x = self.randwire_3(x)

        x = self.relu(x)
        x = self.conv_out(x)
        x = self.batch_norm_out(x)

        x = self.global_average_pool(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x
