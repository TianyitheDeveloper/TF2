import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers,models,regularizers,constraints,losses,metrics

# tf.keras.backend.clear_session()
# model = models.Sequential()
# model.add(layers.Dense(64,input_dim=64,
#                        kernel_regularizer=regularizers.l2(0.01),
#                        activity_regularizer=regularizers.l1(0.01),
#                        kernel_constraint=constraints.MaxNorm(max_value=2,axis=0)))
# model.add(layers.Dense(10,
#                        kernel_regularizer=regularizers.l1_l2(0.01,0.02),
#                        activation="sigmoid"))
# model.compile(optimizer='rmsprop',loss="sparse_categorical_crossentropy",metrics=["AUC"])
# model.summary()


# # 自定义损失函数
# class FocalLoss(losses.Loss):
#     def __init__(self,gamma=2.0,alpha=0.25):
#         self.gamma = gamma
#         self.alpha = alpha
#
#     def call(self,y_true,y_pred):
#
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         loss = -tf.sum(self.alpha * tf.pow(1. - pt_1, self.gamma) * tf.log(1e-07+pt_1)) \
#            -tf.sum((1-self.alpha) * tf.pow( pt_0, self.gamma) * tf.log(1. - pt_0 + 1e-07))
#         return loss
#
#
# c=FocalLoss()
# c.call([0],[1])

# 自定义损失函数
# @tf.function()
# def ks(y_true,y_pred):
#     y_true = tf.reshape(y_true, (-1,))
#     y_pred = tf.reshape(y_pred, (-1,))
#     length = tf.shape(y_true)[0]
#     t = tf.math.top_k(y_pred,k=length,sorted=False)
#     y_pred_sorted = tf.gather(y_pred,t.indices)
#     y_true_sorted = tf.gather(y_true, t.indices)
#     cum_positive_ratio = tf.truediv(tf.cumsum(y_true_sorted), tf.reduce_sum(y_true_sorted))
#     cum_negetive_ratio = tf.truediv(tf.cumsum(1-y_true_sorted), tf.reduce_sum(1-y_true_sorted))
#     ks_value = tf.reduce_max(tf.abs(cum_positive_ratio-cum_negetive_ratio))
#     return ks_value
#
# y_true = tf.constant([[1],[1],[1],[0],[1],[1],[1],[0],[0],[0],[1],[0],[1],[0]])
# y_pred = tf.constant([[0.6],[0.1],[0.4],[0.5],[0.7],[0.7],[0.7],
#                       [0.4],[0.4],[0.5],[0.8],[0.3],[0.5],[0.3]])
# tf.print(ks(y_true,y_pred))

# class KS(metrics.Metric):
#
#     def __init__(self, name = "ks", **kwargs):
#         super(KS,self).__init__(name=name,**kwargs)
#         self.true_positives = self.add_weight(
#             name = "tp",shape = (101,), initializer = "zeros")
#         self.false_positives = self.add_weight(
#             name = "fp",shape = (101,), initializer = "zeros")
#
#     @tf.function
#     def update_state(self,y_true,y_pred):
#         y_true = tf.cast(tf.reshape(y_true,(-1,)),tf.bool)
#         y_pred = tf.cast(100*tf.reshape(y_pred,(-1,)),tf.int32)
#
#         for i in tf.range(0,tf.shape(y_true)[0]):
#             if y_true[i]:
#                 self.true_positives[y_pred[i]].assign(
#                     self.true_positives[y_pred[i]]+1.0)
#             else:
#                 self.false_positives[y_pred[i]].assign(
#                     self.false_positives[y_pred[i]]+1.0)
#         return (self.true_positives,self.false_positives)
#
#     @tf.function
#     def result(self):
#         cum_positive_ratio = tf.truediv(
#             tf.cumsum(self.true_positives),tf.reduce_sum(self.true_positives))
#         cum_negative_ratio = tf.truediv(
#             tf.cumsum(self.false_positives),tf.reduce_sum(self.false_positives))
#         ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
#         return ks_value
# y_true = tf.constant([[1],[1],[1],[0],[1],[1],[1],[0],[0],[0],[1],[0],[1],[0]])
# y_pred = tf.constant([[0.6],[0.1],[0.4],[0.5],[0.7],[0.7],
#                       [0.7],[0.4],[0.4],[0.5],[0.8],[0.3],[0.5],[0.3]])
#
# myks = KS()
# myks.update_state(y_true,y_pred)
# tf.print(myks.result())

# #打印时间分割线
# @tf.function
# def printbar():
#     ts = tf.timestamp()
#     today_ts = ts%(24*60*60)
#
#     hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
#     minite = tf.cast((today_ts%3600)//60,tf.int32)
#     second = tf.cast(tf.floor(today_ts%60),tf.int32)
#
#     def timeformat(m):
#         if tf.strings.length(tf.strings.format("{}",m))==1:
#             return(tf.strings.format("0{}",m))
#         else:
#             return(tf.strings.format("{}",m))
#
#     timestring = tf.strings.join([timeformat(hour),timeformat(minite),
#                 timeformat(second)],separator = ":")
#     tf.print("========"*8,end = "")
#     tf.print(timestring)
#
# # 求f(x) = a*x**2 + b*x + c的最小值
# # 使用optimizer.apply_gradients
# x = tf.Variable(0.0,name = "x",dtype = tf.float32)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
# @tf.function
# def minimizef():
#     a = tf.constant(1.0)
#     b = tf.constant(-2.0)
#     c = tf.constant(1.0)
#
#     while tf.constant(True):
#         with tf.GradientTape() as tape:
#             y = a*tf.pow(x,2) + b*x + c
#         dy_dx = tape.gradient(y,x)
#         optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
#
#         #迭代终止条件
#         if tf.abs(dy_dx)<tf.constant(0.00001):
#             break
#
#         if tf.math.mod(optimizer.iterations,100)==0:
#             printbar()
#             tf.print("step = ",optimizer.iterations)
#             tf.print("x = ", x)
#             tf.print("")
#
#     y = a*tf.pow(x,2) + b*x + c
#     return y
#
# tf.print("y =",minimizef())
# tf.print("x =",x)
#
# # 求f(x) = a*x**2 + b*x + c的最小值
# # 使用optimizer.minimize
# x = tf.Variable(0.0,name = "x",dtype = tf.float32)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
# def f():
#     a = tf.constant(1.0)
#     b = tf.constant(-2.0)
#     c = tf.constant(1.0)
#     y = a*tf.pow(x,2)+b*x+c
#     return(y)
#
# @tf.function
# def train(epoch = 1000):
#     for _ in tf.range(epoch):
#         optimizer.minimize(f,[x])
#     tf.print("epoch = ",optimizer.iterations)
#     return(f())
#
# train(1000)
# tf.print("y = ",f())
# tf.print("x = ",x)
#
# # 求f(x) = a*x**2 + b*x + c的最小值
# # 使用model.fit
# tf.keras.backend.clear_session()
# class FakeModel(tf.keras.models.Model):
#     def __init__(self,a,b,c):
#         super(FakeModel,self).__init__()
#         self.a = a
#         self.b = b
#         self.c = c
#
#     def build(self):
#         self.x = tf.Variable(0.0,name = "x")
#         self.built = True
#
#     def call(self,features):
#         loss  = self.a*(self.x)**2+self.b*(self.x)+self.c
#         return(tf.ones_like(features)*loss)
#
# def myloss(y_true,y_pred):
#     return tf.reduce_mean(y_pred)
#
# model = FakeModel(tf.constant(1.0),tf.constant(-2.0),tf.constant(1.0))
#
# model.build()
# model.summary()
#
# model.compile(optimizer =
#               tf.keras.optimizers.SGD(learning_rate=0.01),loss = myloss)
# history = model.fit(tf.zeros((100,2)),
#                     tf.ones(100),batch_size = 1,epochs = 10)  #迭代1000次
# tf.print("x=",model.x)
# tf.print("loss=",model(tf.constant(0.0)))









tf.print("success")