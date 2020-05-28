import tensorflow as tf
import numpy as np

# # 标量运算
# a = tf.constant([[10.0,2],[-3,4.0]])
# b = tf.constant([[5.0,6],[7,8.0]])
# tf.print(a+b)   # 对应位置相加
# tf.print(tf.add(a,b))   # 与上一种方法类似
# tf.print(tf.add_n([a,b]))   # 多个值相加
# tf.print(a*b)   # 对应位置相乘
# tf.print(a/b)   # 对应位置相除
# tf.print(a**2)  # 与乘法类似
# tf.print(a**0.5)        # 对负数也能开方
# tf.print(tf.sqrt(a))    # 与上一种方法不同
# tf.print(a%3)
# tf.print(a//3)
# tf.print(a>=2)
# tf.print((a<=3)&(a>=2)) # 此处不应用and，而应用&
# tf.print((a<=3)|(a>=2)) # 此处不应用or，而应用|
# tf.print(a==2)
# tf.print(tf.maximum(a,b))   # 取每个位置处的最大值

# # 向量运算
# a = tf.range(1,10)
# tf.print(a)
# tf.print(tf.reduce_sum(a))      # 连加
# tf.print(tf.reduce_mean(a))     # 均值
# tf.print(tf.reduce_max(a))      # 一个向量中的最大值
# tf.print(tf.reduce_prod(a))     # 连乘
# b = tf.reshape(a,[3,3])
# tf.print(b)
# tf.print(tf.reduce_sum(b,axis=1,keepdims=True)) # 将每行的结果加起来
# tf.print(tf.reduce_sum(b,axis=0,keepdims=True)) # 将每列的结果加起来
# p = tf.constant([True,False,False])
# q = tf.constant([False,False,True])
# tf.print(tf.reduce_all(p))      # and
# tf.print(tf.reduce_any(q))      # or
# tf.print(tf.foldr(lambda a,b:a+b,tf.range(10))) # 实现tf.folder实现tf.reduce_sum
# tf.print(tf.math.cumsum(a))     # 扫描连加
# tf.print(tf.math.cumprod(a))    # 扫描连乘
# tf.print(tf.argmax(a))          # 最大值索引
# tf.print(tf.argmin(a))          # 最小值索引
# tf.print(tf.math.top_k(a,3,sorted=False))       # 最大的3个值

# # 矩阵运算
# a = tf.constant([[1,2],[3,4]])
# b = tf.constant([[2,0],[0,2]])
# tf.print(a@b)                       # 两个矩阵相乘
# tf.print(tf.transpose(a))           # 矩阵转置
# tf.print(tf.linalg.trace(a))        # 求迹
# tf.print(tf.linalg.norm(tf.cast(a,tf.float32)))         # 求范数（必须使用float型）
# tf.print(tf.linalg.det(tf.cast(a,tf.float32)))          # 求行列式（必须使用float型）
# tf.print(tf.linalg.eigvalsh(tf.cast(a,tf.float32)))     # 求特征值
# tf.print(tf.linalg.qr(tf.cast(a,tf.float32)))           # 矩阵分解
# tf.print(tf.linalg.qr(tf.cast(a,tf.float32))[0]@tf.linalg.qr(tf.cast(a,tf.float32))[1])
# tf.print(tf.linalg.svd(tf.cast(a,tf.float32))[0])       # SVD分解
# tf.print(tf.linalg.svd(tf.cast(a,tf.float32))[1])       # SVD分解
# tf.print(tf.linalg.svd(tf.cast(a,tf.float32))[2])       # SVD分解
# tf.print(tf.linalg.svd(tf.cast(a,tf.float32))[1]@tf.linalg.diag(tf.linalg.svd(tf.cast(a,tf.float32))[0])@tf.linalg.svd(tf.cast(a,tf.float32))[2])

# # 广播机制
# a = tf.constant([1,2,3])
# b = tf.constant([[0,0,0],[1,1,1],[2,2,2]])
# tf.print(a+b)
# tf.print(b+a)
# tf.print(tf.broadcast_to(a,b.shape))                    # 把a变成b的形状
# tf.print(tf.broadcast_static_shape(a.shape,b.shape))    # 静态形状，Tensorshape类型参数
# c = tf.constant([1,2,3])
# d = tf.constant([[1],[2],[3]])
# tf.print(tf.broadcast_dynamic_shape(tf.shape(c),tf.shape(d)))   # 动态形状，Tensor类型参数
# tf.print(c+d)

# @tf.function
# def np_random():
#     a = np.random.randn(3,3)
#     tf.print(a)
#
# @tf.function
# def tf_random():
#     a = tf.random.normal((3,3))
#     tf.print(a)
#
# for _ in range(3):
#     np_random()     # 运行三次，每次都相同，假随机
# for _ in range(3):
#     tf_random()     # 运行三次，每次都不同，真随机

# x = tf.Variable(1.0,dtype=tf.float32)
# @tf.function
# def outer_var():
#     x.assign_add(1.0)
#     tf.print(x)
#     return x
#
# for _ in range(3):
#     outer_var()

# tensor_list = []
# # 此处不可添加@tf.funtion
# def append_tensor(x):
#     tensor_list.append(x)
#     return tensor_list
#
# append_tensor(tf.constant(5.0))
# append_tensor(tf.constant(6.0))
# tf.print(tensor_list)

# @tf.function(autograph=True)
# def myadd(a,b):
#     for i in tf.range(3):
#         tf.print(i)
#     c = a+b
#     print("tracing")
#     return c
# # 第一次调用这个被@tf.function装饰的函数，参数是Tensor
# myadd(tf.constant("hello"),tf.constant("world"))
# # 第二次调用这个被@tf.function装饰的函数，参数是Tensor
# myadd(tf.constant("hello"),tf.constant("TF2"))
# # 第二次调用这个被@tf.function装饰的函数，但参数的类型发生变化
# myadd(tf.constant(1),tf.constant(1))
# # 第一次调用这个被@tf.function装饰的函数，参数不是Tensor
# myadd(1,2)
# # 第一次调用这个被@tf.function装饰的函数，参数不是Tensor
# myadd("hello","TF2")

# x = tf.Variable(1.0,dtype=tf.float32)
# # 在tf.function中用input——signature限定输入张量的签名类型：shape和dtype
# @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
# def add_float(a):
#     x.assign_add(a)
#     tf.print(x)
#     return x
#
# add_float(tf.constant(3.0))
# # add_float(tf.constant(1))     # 会报错
# add_float(3.0)
# add_float(1)

# class DemoModule(tf.Module):
#     def __init__(self,init_value = tf.constant(0.0),name=None):
#         super(DemoModule, self).__init__(name=name)
#         with self.name_scope:
#             self.x = tf.Variable(init_value,dtype=tf.float32,trainable=True)
#
#     @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
#     def add_float(self,a):
#         with self.name_scope:
#             self.x.assign_add(a)
#             tf.print(self.x)
#             return self.x
#
# demo = DemoModule(init_value=tf.constant(1.0))
# result = demo.add_float(tf.constant(5.0))
# tf.print(demo.variables)
# tf.print(demo.trainable_variables)
# tf.print(demo.submodules)
# # tf.saved_model.save(demo,"data/demo/1",signatures={"serving_default":demo.add_float})
#
# demo2 = tf.saved_model.load("data/demo/1")
# demo2.add_float(tf.constant(5.0))

# mymodule = tf.Module()
# mymodule.x = tf.Variable(0.0)
#
# @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
# def add_float(a):
#     mymodule.x.assign_add(a)
#     tf.print(mymodule.x)
#     return mymodule.x
#
# mymodule.add_float = add_float
# tf.print(mymodule.add_float(tf.constant(1.0)).numpy())
#
# tf.saved_model.save(mymodule,"data/mymodule",signatures={"serving_default":mymodule.add_float})
# mymodule2 = tf.saved_model.load("data/mymodule")
# mymodule2.add_float(tf.constant(5.0))

from tensorflow.keras import models,layers,losses,metrics
tf.print(issubclass(tf.keras.Model,tf.Module))
tf.print(issubclass(tf.keras.layers.Layer,tf.Module))
tf.print(issubclass(tf.keras.Model,tf.keras.layers.Layer))
tf.keras.backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(4,input_shape=(10,)))
model.add(layers.Dense(2))
model.add(layers.Dense(1))
tf.print(model.trainable_variables)
model.layers[0].trainable = False   # 冻结第0层的变量，使之不可训练
tf.print(model.trainable_variables)
tf.print(model.submodules)          # 模型的子单元
tf.print(model.name)
tf.print(model.name_scope())





tf.print("success")