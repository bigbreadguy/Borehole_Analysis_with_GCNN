import tensorflow
from delaynay2D import *
from preprocess import *
from model import *


train_data, eval_data, test_data = preprocess(file="data.xlsx")


EPOCHS = 10

n_epoch = 0
model = GCNN()
while n_epoch < EPOCHS:
    for d0, d1 in zip(*[iter(train_data)*2]):
        train_loss = []
        x = tf.convert_to_tensor([d0["node_features"],d1["node_features"]])
        a = tf.convert_to_tensor([d0["adjacency_matrix"],d1["adjacency_matrix"]])
        y = tf.convert_to_tensor([d0["target"],d1["target"]])

        model.update(a)
        optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        pred, loss = train_step2(model,x,a,y)
        train_loss.append(loss)

    #print("prediction : ", pred)
    print("train loss : ", tf.math.sqrt(tf.reduce_mean(train_loss)))


    n_epoch += 1
