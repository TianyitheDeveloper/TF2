import tensorflow as tf
from tensorflow.keras import layers,losses,metrics,optimizers,models
import pandas as pd
import random
# 使用TF的低阶API实现线性回归
# 打印时间分割线
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts%(24*60*60)
    hour = tf.cast(today_ts//3600 +8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m)) == 1:
            return tf.strings.format("0{}",m)
        else:
            return tf.strings.format("{}",m)

    timrstring = tf.strings.join([timeformat(hour),timeformat(minite),timeformat(second)],separator=":")
    tf.print("==="*20,end="")
    tf.print(timrstring)

# 样本数量
n = 1000
# 测试用数据集
X0 = tf.random.uniform([n,1],minval=-5,maxval=5)
X1 = tf.random.uniform([n,1],minval=-10,maxval=0)
X2 = tf.random.uniform([n,1],minval=0,maxval=10)
w0 = tf.constant([[3.0]])
w1 = tf.constant([[7.0]])
w2 = tf.constant([[-2.0]])
w3 = tf.constant([[-8.0]])
w4 = tf.constant([[5.0]])
w5 = tf.constant([[9.0]])
b0 = tf.constant(15.0)
Y =X0@w0 +X1@w1 +X2@w2 + X0*X1@w3 + X0*X2@w4 + X1*X2@w5 + b0 + tf.random.normal([n,1],mean=0.0,stddev=2.0)

X = tf.concat([X0,X1,X2],axis=1)
dataX = pd.DataFrame(X)
dataY = pd.DataFrame(Y)
data = pd.concat([dataX,dataY],axis=1)
tf.print(data.shape)

data.to_excel("POLY.xlsx")



# # 使用动态图调试
# w = tf.Variable(tf.random.normal(w0.shape))
# b = tf.Variable(0.0)
# # 使用autograph机制转换成静态图（速度会变得巨快）
# @tf.function
# def train(epoches):
#     for epoch in tf.range(1,epoches+1):
#         with tf.GradientTape() as tape:
#             # 正向传播求损失
#             Y_hat = X@w + b
#             loss = tf.squeeze(tf.transpose(Y-Y_hat)@(Y-Y_hat))/(2.0*n)
#         # 反向传播求梯度(仅仅是求一个导数而已)
#         # dloss_dw, dloss_db = tape.gradient(loss,[w,b])
#         dloss_dw = tf.reduce_sum(tf.transpose(Y_hat-Y)@X)/n
#         dloss_db = tf.reduce_sum(Y_hat-Y)/n
#         # 梯度下降更新参数(这里是利用刚刚的导数来更新参数)
#         w.assign(w-0.002*dloss_dw)
#         b.assign(b-0.002*dloss_db)
#         if epoch%1000 == 0:
#             printbar()
#             tf.print("epoch=",epoch,"loss=",loss)
#             tf.print("w=",w)
#             tf.print("b=",b)
#             tf.print("")
#
# train(50000)


# # 构建输入数据管道
# ds = tf.data.Dataset.from_tensor_slices((X,Y)).shuffle(buffer_size=1000).batch(100).prefetch(tf.data.experimental.AUTOTUNE)
# # 定义优化器
# optimizer = optimizers.SGD(learning_rate=0.001)
#
# liner = layers.Dense(units=1)
# liner.build(input_shape=(2,))
# @tf.function
# def train(epoches):
#     for epoch in tf.range(1,epoches+1):
#         L = tf.constant(0.0)
#         for X_batch,Y_batch in ds:
#             with tf.GradientTape() as tape:
#                 Y_hat = liner(X_batch)
#                 loss = losses.mean_squared_error(tf.reshape(Y_hat,[-1]),tf.reshape(Y_batch,[-1]))
#             grads = tape.gradient(loss,liner.variables)
#             optimizer.apply_gradients((zip(grads,liner.variables)))
#             L = loss
#         if epoch%100 == 0:
#             printbar()
#             tf.print("epoch=",epoch,"loss=",L)
#             tf.print("w=",liner.kernel)
#             tf.print("b=",liner.bias)
#             tf.print()

# tf.keras.backend.clear_session()
# linear = models.Sequential()
# linear.add(layers.Dense(1,input_shape=(2,)))
# linear.compile(optimizer="adam",loss="mse",metrics=["mae"])
# linear.fit(X,Y,batch_size=20,epochs=200)
# tf.print("w=",linear.layers[0].kernel)
# tf.print("b=",linear.layers[0].bias)

tf.print("success")