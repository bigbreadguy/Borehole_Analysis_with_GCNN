import tensorflow as tf #TensorFlow 2.0(and over) is required.

class ParallelDenseLayer(tf.keras.layers.Layer):

  def __init__(self, units=64):
    super(ParallelDenseLayer, self).__init__()
    self.units = units

  def call(self, x):
    return tf.stack([tf.keras.layers.Dense(units=self.units)(x[:,i,:]) for i in range(x.shape[-2])], axis=1)

class GraphConvolutionLayer(tf.keras.layers.Layer):

  def __init__(self, adjacency_matrix, units=64):
    super(GraphConvolutionLayer, self).__init__()
    self.adjacency_matrix = adjacency_matrix
    self.units = units

  def build(self, input_shape):
    self.a_weight = self.add_weight(
        name='a_weight',
        shape=self.adjacency_matrix.shape,
        initializer='ones',
        trainable=True)
    self.a_bias = self.add_weight(
        name='a_bias',
        shape=self.adjacency_matrix.shape,
        initializer='zeros',
        trainable=True)

  def call(self, x):
    self.adjacency_matrix.assign(self.a_weight * self.adjacency_matrix + self.a_bias)
    return tf.linalg.matmul(self.adjacency_matrix, x)
  
class GraphConvolutionBlock(tf.keras.Model):
  def __init__(self, adjacency_matrix, units=64):
    super(GraphConvolutionBlock, self).__init__()
    self.adjacency_matrix = adjacency_matrix
    self.units = units

    self.d_inv = np.sqrt(np.array([np.diag(i) for i in np.sum(self.adjacency_matrix, axis=1)])**-0.5)
    self.d_inv[self.d_inv == np.inf] = 0.0

    self.conv1d = tf.keras.layers.Conv1D(filters=self.units, kernel_size=15, padding='same')
    self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

  def call(self, x):
    x = self.conv1d(x)
    x = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.matmul(self.d_inv, self.adjacency_matrix), self.d_inv), x)
    x = self.bn(x)
    return tf.nn.relu(x)
    
class GCNN(tf.keras.Model):
  def __init__(self):
    super(GCNN, self).__init__()
    self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
    #self.adjacency_matrix = tf.Variable(0., shape=tf.TensorShape(None))
  
  def update(self, adjacency_matrix):
    self.adjacency_matrix = adjacency_matrix

  def loss(self, y, pred):
    return tf.keras.losses.MeanSquaredError()(y, pred)

  def call(self, x):
    a = self.adjacency_matrix
    x = tf.keras.layers.Dense(16)(x)
    x = GraphConvolutionBlock(a, units=64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = GraphConvolutionBlock(a, units=64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = GraphConvolutionBlock(a, units=64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    return x
