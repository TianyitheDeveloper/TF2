import numpy as np
import tensorflow as tf
import datetime

# i = tf.constant(1)                  # 常量i，值为1
# l = tf.constant(1,dtype=tf.int64)   # 常量l，值为1，datatype为int64
# k = tf.constant(True)               # 常量k，值为True
# h = tf.constant([1,2,3])            # 常量h，值为一维数组[1,2,3]
# j = tf.constant([[1,2],[3,4]])      # 常量j，值为二维数组[[1,2],[3,4]]

# print(tf.rank(j))                   # 输出常量l的rank
# print(j.numpy().ndim)               # numpy的ndim与tf.rank用法一致
# print(np.ndim(j))                   # 或用np的形式

# print(h.dtype,tf.cast(h,tf.float32).dtype)      # 使用tf.cast将数据类型改掉
# print(h.shape)

# c = tf.constant([1.,2.])            # 定义常量
# print(c)                            # 查看c的值
# print(id(c))                        # 查看常量c的地址
# c = c + tf.constant([1.,1.])        # 常量的重新赋值
# print(c)                            # 查看c的值
# print(id(c))                        # 再次查看c的地址


# v = tf.Variable([1.0,2.0],name="v") # 定义变量
# print(v)                            # 输出变量
# print(id(v))                        # 查看地址
# v.assign_add([1.,1.])               # 加值
# print(v)                            # 再次输出
# print(id(v))                        # 再次查看地址

# x = tf.constant("hello")
# y = tf.constant("TF2")
#
# @tf.function
# def strjoin(x,y):
#     """
#     使用autograph构建静态图
#     :param x:
#     :param y:
#     :return:
#     """
#     z = tf.strings.join([x,y],separator=" ")
#     tf.print(z)
#     return z

# result = strjoin(x,y)
# print(result)

# x = tf.Variable(0.0, name="x",dtype=tf.float32)    # 设置x=0
# a = tf.constant(1.0)                               # 设置a=1
# b = tf.constant(-2.0)                              # 设置b=-2
# c = tf.constant(1.0)                               # 设置c=1
#
# with tf.GradientTape() as tape:
#     tape.watch([a,b,c])                            # 对常量张量也可以求导，但需添加watch
#     y = a*tf.pow(x,2) + b*x +c                     # 设置y=ax²+bx+c
# dy_dx,dy_da = tape.gradient(y,[x,a])               # 计算梯度
# tf.print(dy_dx)
# tf.print(dy_da)
#
# with tf.GradientTape() as tape2:
#     with tf.GradientTape() as tape1:
#         y = a*tf.pow(x,2) + b*x +c                 # 设置y=ax²+bx+c
#     dy_dx,dy_da = tape1.gradient(y,[x,a])          # 计算一阶导数
# dy2_dx2 = tape2.gradient(dy_dx,x)                  # 计算二阶导数
# tf.print(dy2_dx2)

# # 可在autograph中使用
# @tf.function
# def fun(x):
#     a = tf.constant(1.0)  # 设置a=1
#     b = tf.constant(-2.0)  # 设置b=-2
#     c = tf.constant(1.0)  # 设置c=1
#     x = tf.cast(x,tf.float32)   # 不管自变量是什么数据类型，都将之转换成tf.float32
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         y = a*tf.pow(x,2) + b*x +c
#     dy_dx = tape.gradient(y,x)
#     return (dy_dx,y)
#
# tf.print(fun(tf.constant(0.0)))
# tf.print(fun(tf.constant(1.0)))

# # 使用梯度磁带和优化器求最小值
# x = tf.Variable(0.0, name="x",dtype=tf.float32)    # 设置x=0
# a = tf.constant(1.0)                               # 设置a=1
# b = tf.constant(-2.0)                              # 设置b=-2
# c = tf.constant(1.0)                               # 设置c=1
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) # 定义优化器
# for _ in range(1000):
#     with tf.GradientTape() as tape:
#         y = a*tf.pow(x,2) + b*x +c
#     dy_dx = tape.gradient(y, x)
#     optimizer.apply_gradients(grads_and_vars=[[dy_dx,x]])   # 可以对多个未知量进行梯度下降
# tf.print("y=",y,"x=",x)

# 在autograph中完成最小值，使用optimizer.minimize
x = tf.Variable(0.0,name="x",dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def fun():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2) + b*x +c
    return y

@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(fun,[x])
    return (fun())

tf.print(train(1000))
tf.print(x)











print('success')