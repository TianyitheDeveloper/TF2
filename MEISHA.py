import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('POLY.xlsx')

# sns.heatmap(data.corr())
# plt.show()

X = tf.constant(data.drop('y',axis=1).values,dtype=tf.float32,shape=[1000,3])
Y = tf.constant(data['y'],dtype=tf.float32,shape=[1000,1])
train_X = X[:900]
train_Y = Y[:900]
test_X = X[900:]
test_Y = Y[900:]

#定义参数
w0 = tf.Variable(tf.random.normal([3,1]))
w3 = tf.Variable(tf.random.normal([1,1]))
w4 = tf.Variable(tf.random.normal([1,1]))
w5 = tf.Variable(tf.random.normal([1,1]))
w6 = tf.Variable(tf.random.normal([1,1]))
b = tf.Variable(0.0)

# 训练
@tf.function
def train(epoches):
    for epoch in tf.range(1,epoches+1):
        tf.random.shuffle(X)
        X_s = train_X[:900]
        Y_s = train_Y[:900]

        X_s0 = tf.reshape(X_s[:, 0], [900, 1])
        X_s1 = tf.reshape(X_s[:, 1], [900, 1])
        X_s2 = tf.reshape(X_s[:, 2], [900, 1])
        with tf.GradientTape() as tape:
            Y_hat =X_s@w0 + X_s0*X_s1@w3 + X_s0*X_s2@w4 + X_s1*X_s2@w5 +X_s1*X_s2*X_s0@w6 + b
            # Y_hat = X_s@k + b
            loss = tf.squeeze(tf.transpose(Y_hat-Y_s)@(Y_hat-Y_s))/(2.0*900)

        dloss_dw0,dloss_dw3,dloss_dw4,dloss_dw5,dloss_dw6, dloss_db = tape.gradient(loss,[w0,w3,w4,w5,w6,b])

        w0.assign(w0 - 0.0002 * dloss_dw0)
        w3.assign(w3 - 0.0002 * dloss_dw3)
        w4.assign(w4 - 0.0002 * dloss_dw4)
        w5.assign(w5 - 0.0002 * dloss_dw5)
        w6.assign(w6 - 0.0002 * dloss_dw6)
        b.assign(b - 0.0002 * dloss_db)

        if epoch%50000 == 0:
            tf.print("epoch=",epoch,"loss=",loss)
            tf.print("w0=", w0)
            tf.print("w3=", w3)
            tf.print("w4=", w4)
            tf.print("w5=", w5)
            tf.print("w6=", w6)
            tf.print("b=", b)
            tf.print("")


train(500000)



tf.print("success")



