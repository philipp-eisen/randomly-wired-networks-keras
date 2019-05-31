import tensorflow as tf
from tensorflow.python import keras
import networkx as nx


class WeightedSum(keras.layers.Layer):
    def __init__(self, kernel_regularizer=None, **kwargs):
        self.kernel_regularizer = kernel_regularizer
        self.aggregate_w = None
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        if type(input_shape) is not list:
            raise TypeError("A node needs to be called/build with a list of "
                            "tensors (even if it only has one parent node)")
        if not all([shape.is_compatible_with(input_shape[0]) for shape in input_shape]):
            raise ValueError("All inputs must have compatible input shapes. Found {}".format(input_shape))

        shape = tf.TensorShape(len(input_shape))

        self.aggregate_w = self.add_weight(
            name='{}_aggregate_w'.format(self.name),
            shape=shape,
            initializer=keras.initializers.zeros,
            trainable=True,
            regularizer=self.kernel_regularizer
        )

    def call(self, inputs, **kwargs):
        x = tf.stack(inputs)
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
                 dropout_rate=0.2,
                 **kwargs):
        """
        Args:
            channels: The number of channels (filters).
            kernel_size: The size of the convolution kernel. Defaults to `(3, 3)`
            strides: The strides used for the convolution operation. Defaults to `(1, 1)`
            padding: one of `"valid"` or `"same"` (case-insensitive). Defaults to `"same"`
            kernel_regularizer: Optional regularizer for the convolution kernel as well as the `WeightedSum`
                                sum operation.
            bias_regularizer: Optional regularizer for the bias vector.
            **kwargs: passed through to the `SeparableConv2D` operation.
        """
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
            bias_regularizer=self.bias_regularizer,
            **kwargs
        )

        self.batch_norm = keras.layers.BatchNormalization(
            name='{}_batch_norm'.format(self.name)
        )
        self.drop_out = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.weighted_sum(inputs)

        x = self.relu(x)
        x = self.conv(x, **kwargs)
        x = self.batch_norm(x)
        x = self.drop_out(x)
        return x


class RandomWiring(keras.Model):
    def __init__(self,
                 channels,
                 n,
                 random_graph_algorithm,
                 strides=(2, 2),
                 kernel_size=(3, 3),
                 k=4,
                 p=0.75,
                 m=5,
                 seed=0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        """
        Args:
            channels (int): The number of channels all convolution operations in this `RandomWiring` stage produce.
            n (int): int: The number of nodes in this `RandomWiring` stage.
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
                    - `p`: protbability of edge creation
            strides:
            kernel_size (tuple(int, int)):
            k (int): refer to "`random_graph_algorithm`" for explanation of the argument. Defaults to
                best performing value in arxiv 1904.01569.
            p (float): refer to "`random_graph_algorithm`" for explanation of the argument. Defaults to
                best performing value in arxiv 1904.01569.
            m (int): refer to "`random_graph_algorithm`" for explanation of the argument. Defaults to
                best performing value in arxiv 1904.01569.
            seed (int): int. Seed with which the random graph algorithms are seeded. Defaults to fixed `0`.
            kernel_regularizer (keras.regularizers.Regularizer): applied to all kernel weights of all nodes
            bias_regularizer (keras.regularizers.Regularizer): applied to all bias nodes of all nodes
            **kwargs:
        """
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
            raise ValueError('`random_graph_model` must be either "ws", "ba" or "er"')

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
