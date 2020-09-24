import tensorflow
from delaynay2D import *
from preprocess import *
from model import *


train_data, eval_data, test_data = preprocess(file="data.xlsx")

model = GCNN()

test_loss = []
for d0, d1 in zip(*[iter(test_data)]*2):
    x = tf.convert_to_tensor([d0["node_features"],d1["node_features"]])
    a = tf.convert_to_tensor([d0["adjacency_matrix"],d1["adjacency_matrix"]])
    y = tf.convert_to_tensor([d0["target"],d1["target"]])

    model.update(a)
    pred = model(x, training=False)
    loss = model.loss(y, pred)
    test_loss.append(loss)

tf.math.sqrt(tf.reduce_mean(test_loss))
