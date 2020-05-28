import tensorflow as tf
import numpy as np

# # 创建张量
# tf.print(tf.constant([1,2,3],dtype=tf.float32))     # 常量
# tf.print(tf.range(1,10,delta=2))                    # range范围
# tf.print(tf.linspace(0.0,2*3.14,5))                 # linspace范围
# tf.print(tf.zeros([3,3]))                           # 全0矩阵
# tf.print(tf.one_hot(1,5))                           # onehot向量
# tf.print(tf.ones([3,3]))                            # 全1矩阵
# tf.print(tf.zeros_like(tf.ones([3,3])))             # 与某矩阵相同大小的全0矩阵
# tf.print(tf.fill([3,2],5))                          # 用某值全部填充某矩阵
# tf.random.set_seed(1.0)
# # 随机生成一个元素最大值为10，最小值为0的矩阵
# tf.print(tf.random.uniform([5,2],minval=0,maxval=10))
# # 随机生成一个元素服从0均值1方差正态分布的矩阵
# tf.print(tf.random.normal([3,3],mean=0.0,stddev=1))
# # 随机生成一个元素服从0均值1方差正态分布的矩阵，正态分布随机，剔除两倍方差以外数据重新生成
# tf.print(tf.random.truncated_normal([4,4],mean=0.0,stddev=1.0,dtype=tf.float32))
# tf.print(tf.eye(3,3))                               # 单位矩阵
# tf.print(tf.linalg.diag([1,2,3]))                   # 对角阵

# # 索引切片
# tf.random.set_seed(3)
# t = tf.random.uniform([5,5],minval=0,maxval=10,dtype=tf.int32)
# # 输出全部
# tf.print(t)
# # 输出第一行
# tf.print(t[0])
# # 输出倒数第一行
# tf.print(t[-1])
# # 输出第二行第三列位置的数
# tf.print(t[1,2])
# tf.print(t[1][2])
# # 输出第1行至第3行的数据
# tf.print(t[:3,:])
# # 每隔一行取一行，每个一列取一列
# tf.print(t[::2,::2])
# # 对于变量，可使用索引和切片修改部分元素
# x = tf.Variable([[1,2],[3,4]],dtype=tf.float32)
# x[1,:].assign(tf.constant([0.0,0.0]))
# tf.print(x)
# # 使用tf.boolean_mask实现布尔索引
# # 找到矩阵中小于0的元素
# c = tf.constant([[1,-1,1],[2,3,-1],[-2,-3,0]],dtype=tf.int32)
# tf.print(c)
# tf.print(c[c<0])
# tf.print(tf.boolean_mask(c,c<0))
# # 找到张量中小于0的元素，转换其值
# tf.print(tf.where(c<0,tf.fill(c.shape,0),c))
# # 若where只有一个参数，将返回所有满足条件的位置坐标
# tf.print(tf.where(c<0))

# # 维度变换
# a = tf.random.uniform(shape=[1,3,3,2],minval=0,maxval=255,dtype=tf.int32)
# tf.print(a.shape)
# tf.print(a)
# b = tf.reshape(a,[3,6])
# tf.print(b.shape)
# tf.print(b)
# c = tf.reshape(b,[1,3,3,2])
# tf.print(c.shape)
# tf.print(c)
# # 张量在某个维度上只有一个元素，利用tf.squeeze可以消除这个维度
# s = tf.squeeze(a)
# tf.print(s.shape)
# tf.print(s)
# # 在第0维插入长度为1的一个维度
# d = tf.expand_dims(s,axis=0)
# tf.print(d)
# # 通过索引
# a = tf.random.uniform(shape=[1,5,6,4],minval=0,maxval=255,dtype=tf.int32)
# tf.print(a.shape)
# s = tf.transpose(a,perm=[3,1,2,0])
# tf.print(s.shape)

# 合并分隔
a = tf.constant([[1.0,2.0],[3.0,4.0]])
b = tf.constant([[5.0,6.0],[7.0,8.0]])
c = tf.constant([[9.0,10.0],[11.0,12.0]])
# 拼接（不增加维度）
tf.print(tf.concat([a,b,c],axis=0))
# 拼接（不增加维度）
tf.print(tf.concat([a,b,c],axis=1))
# 堆叠（增加维度）
tf.print(tf.stack([a,b,c]))
# 堆叠（增加维度）
tf.print(tf.stack([a,b,c],axis=1))
# 切割
c = tf.concat([a,b,c],axis=0)
tf.print(tf.split(c,3,axis=0))













tf.print("success")