import tensorflow as tf
from tensorflow.python import keras
import networkx as nx


class WeightedSum(keras.layers.Layer):
    def __init__(self, kernel_regularizer=None, **kwargs):
        self.kernel_regularizer = kernel_regularizer
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        if type(input_shape) is not list:
            raise TypeError("A node needs to be called with a list of "
                            "tensors (even if it only has one parent node)")
        if not all([shape.is_compatible_with(input_shape[0]) for shape in input_shape]):
            raise ValueError("All inputs must have the same shape. Found {}".format(input_shape))

        self.aggregate_w = self.add_weight(
            name='{}_aggregate_w'.format(self.name),
            shape=len(input_shape),
            initializer=keras.initializers.zeros,
            trainable=True,
            regularizer=self.kernel_regularizer
        )

    def call(self, inputs, **kwargs):
        x = tf.stack(inputs)
        # Todo: the way I understand the paper even if there is only one input there is a corresponding weight
        x = tf.tensordot(x, keras.backend.sigmoid(self.aggregate_w), [[0], [0]])
        return x

    def get_config(self):
        base_config = super(WeightedSum, self).get_config()
        base_config['kernel_regularizer'] = keras.regularizers.serialize(self.kernel_regularizer)
        return base_config


class Node(keras.Model):
    def __init__(self,
                 channels,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="same",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(Node, self).__init__(**kwargs)

        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.weighted_sum = WeightedSum(kernel_regularizer=kernel_regularizer)

        self.relu = keras.layers.ReLU()

        self.conv = keras.layers.SeparableConv2D(
            name='{}_conv'.format(self.name),
            filters=self.channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        self.batch_norm = keras.layers.BatchNormalization(
            name='{}_batch_norm'.format(self.name)
        )

    def call(self, inputs, training=None, **kwargs):
        x = self.weighted_sum(inputs)

        x = self.relu(x)
        x = self.conv(x, **kwargs)
        x = self.batch_norm(x, training=training)
        return x


class RandomWiring(keras.Model):
    def __init__(self,
                 channels,
                 random_graph_algorithm,
                 strides=(2, 2),
                 kernel_size=(3, 3),
                 n=32,
                 k=4,
                 p=0.75,
                 m=5,
                 seed=1337,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(RandomWiring, self).__init__(**kwargs)

        self.channels = channels
        self.strides = strides
        self.random_graph_model = random_graph_algorithm
        self.kernel_size = kernel_size
        self.n = n
        self.k = k
        self.p = p
        self.seed = seed
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # generate a random graph
        if random_graph_algorithm == "ws":
            random_graph = nx.connected_watts_strogatz_graph(n, k, p, seed=seed)
        elif random_graph_algorithm == "ba":
            random_graph = nx.barabasi_albert_graph(n, m, seed=seed)
        elif random_graph_algorithm == 'er':
            random_graph = nx.erdos_renyi_graph(n, p, seed=seed)
        else:
            raise ValueError('random_graph_model must be either "ws", "ba" or "er"')

        # add the edges as directed edges to a directed graph
        self.dag = nx.DiGraph()
        self.dag.add_edges_from(random_graph.edges)

        self.dag_input_nodes = [node for node in self.dag.nodes if self.dag.in_degree[node] == 0]
        self.dag_output_nodes = [node for node in self.dag.nodes if self.dag.out_degree[node] == 0]

        assert nx.is_directed_acyclic_graph(self.dag)

        self.input_node = Node(channels=self.channels,
                               kernel_size=self.kernel_size,
                               kernel_regularizer=self.kernel_regularizer,
                               bias_regularizer=self.bias_regularizer)

        self.nodes = []
        for node in range(self.n):
            # For the first node in the graph we apply the strides as specified. In the paper this is (2, 2).
            if node in self.dag_input_nodes:
                self.nodes.append(Node(channels=self.channels,
                                       kernel_size=self.kernel_size,
                                       strides=self.strides,
                                       kernel_regularizer=self.kernel_regularizer,
                                       bias_regularizer=self.bias_regularizer))
                continue
            self.nodes.append(Node(channels=self.channels,
                                   kernel_size=self.kernel_size,
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_regularizer=self.bias_regularizer))

        self.output_node = Node(channels=self.channels,
                                kernel_size=self.kernel_size,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer)

    def call(self, inputs, **kwargs):
        node_outputs = [None for _ in range(self.n)]

        x = self.input_node([inputs], **kwargs)
        for node in nx.topological_sort(self.dag):
            if node in self.dag_input_nodes:
                node_outputs[node] = self.nodes[node]([x], **kwargs)
                continue

            predecessor_outputs = []
            for predecessor in self.dag.predecessors(node):
                predecessor_outputs.append(node_outputs[predecessor])

            node_outputs[node] = self.nodes[node](predecessor_outputs, **kwargs)

        dag_outputs = []
        for dag_output_node in self.dag_output_nodes:
            dag_outputs.append(node_outputs[dag_output_node])

        x = self.output_node(dag_outputs, **kwargs)
        return x
