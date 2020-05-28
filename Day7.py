import tensorflow as tf
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers,models,regularizers

# def printlog(info):
#     nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print("\n+"+"="*30+"%s"%nowtime)
#     print(info+'...\n\n')
# # ============================================================
# # 一、构建数据管道
# # ============================================================
# printlog("step1: prepare dataset...")
# dftrain_raw = pd.read_csv("data/titanic/train.csv")
# dftest_raw = pd.read_csv("data/titanic/test.csv")
# dfraw = pd.concat([dftrain_raw,dftest_raw])
#
# def prepare_dfdata(dfraw):
#     dfdata = dfraw.copy()
#     dfdata.columns = [x.lower() for x in dfdata.columns]
#     dfdata = dfdata.rename(columns={'survived':'label'})
#     dfdata = dfdata.drop(['passengerid','name'],axis = 1)
#     for col,dtype in dict(dfdata.dtypes).items():
#         # 判断是否包含缺失值
#         if dfdata[col].hasnans:
#             # 添加标识是否缺失列
#             dfdata[col + '_nan'] = pd.isna(dfdata[col]).astype('int32')
#             # 填充
#             if dtype not in [np.object,np.str,np.unicode]:
#                 dfdata[col].fillna(dfdata[col].mean(),inplace = True)
#             else:
#                 dfdata[col].fillna('',inplace = True)
#     return(dfdata)
#
# dfdata = prepare_dfdata(dfraw)
# dftrain = dfdata.iloc[0:len(dftrain_raw),:]
# dftest = dfdata.iloc[len(dftrain_raw):,:]
#
# # 从 dataframe 导入数据
# def df_to_dataset(df, shuffle=True, batch_size=32):
#     dfdata = df.copy()
#     if 'label' not in dfdata.columns:
#         ds = tf.data.Dataset.from_tensor_slices(dfdata.to_dict(orient = 'list'))
#     else:
#         labels = dfdata.pop('label')
#         ds = tf.data.Dataset.from_tensor_slices((dfdata.to_dict(orient = 'list'), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dfdata))
#     ds = ds.batch(batch_size)
#     return ds
#
# ds_train = df_to_dataset(dftrain)
# ds_test = df_to_dataset(dftest)
#
# # ============================================================
# # 二、定义特征列
# # ============================================================
# printlog("step2:make feature columns...")
#
# feature_columns = []
# # 数值列
# for col in ['age','fare','parch','sibsp'] + [
#     c for c in dfdata.columns if c.endswith('_nan')]:
#     feature_columns.append(tf.feature_column.numeric_column(col))
#
# # 分桶列
# age = tf.feature_column.numeric_column('age')
# age_buckets = tf.feature_column.bucketized_column(age,
#              boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
# feature_columns.append(age_buckets)
#
# # 类别列
# # 注意：所有的Catogorical Column类型最终都要通过indicator_column转换成Dense Column类型才能传入模型！！
# sex = tf.feature_column.indicator_column(
#       tf.feature_column.categorical_column_with_vocabulary_list(
#       key='sex',vocabulary_list=["male", "female"]))
# feature_columns.append(sex)
#
# pclass = tf.feature_column.indicator_column(
#       tf.feature_column.categorical_column_with_vocabulary_list(
#       key='pclass',vocabulary_list=[1,2,3]))
# feature_columns.append(pclass)
#
# ticket = tf.feature_column.indicator_column(
#      tf.feature_column.categorical_column_with_hash_bucket('ticket',3))
# feature_columns.append(ticket)
#
# embarked = tf.feature_column.indicator_column(
#       tf.feature_column.categorical_column_with_vocabulary_list(
#       key='embarked',vocabulary_list=['S','C','B']))
# feature_columns.append(embarked)
#
# # 嵌入列
# cabin = tf.feature_column.embedding_column(
#     tf.feature_column.categorical_column_with_hash_bucket('cabin',32),2)
# feature_columns.append(cabin)
#
# # 交叉列
# pclass_cate = tf.feature_column.categorical_column_with_vocabulary_list(
#           key='pclass',vocabulary_list=[1,2,3])
#
# crossed_feature = tf.feature_column.indicator_column(
#     tf.feature_column.crossed_column([age_buckets, pclass_cate],hash_bucket_size=15))
#
# feature_columns.append(crossed_feature)
#
# # ============================================================
# # 三、定义模型
# # ============================================================
# printlog("step3: define model...")
#
# tf.keras.backend.clear_session()
# model = tf.keras.Sequential([
#   layers.DenseFeatures(feature_columns), #将特征列放入到tf.keras.layers.DenseFeatures中!!!
#   layers.Dense(64, activation='relu'),
#   layers.Dense(64, activation='relu'),
#   layers.Dense(1, activation='sigmoid')
# ])
#
# # ============================================================
# # 四、训练模型
# # ============================================================
# printlog("step4: train model...")
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(ds_train,
#           validation_data=ds_test,
#           epochs=10)
#
# # ============================================================
# # 五、评估模型
# # ============================================================
# printlog("step5: eval model...")
# model.summary()
#
# def plot_metric(history, metric):
#     train_metrics = history.history[metric]
#     val_metrics = history.history['val_'+metric]
#     epochs = range(1, len(train_metrics) + 1)
#     plt.plot(epochs, train_metrics, 'bo--')
#     plt.plot(epochs, val_metrics, 'ro-')
#     plt.title('Training and validation '+ metric)
#     plt.xlabel("Epochs")
#     plt.ylabel(metric)
#     plt.legend(["train_"+metric, 'val_'+metric])
#     plt.show()
#
# plot_metric(history,"accuracy")

# tf.keras.backend.clear_session()
#
# model = models.Sequential()
# model.add(layers.Dense(32,input_shape = (None,16),activation = tf.nn.relu)) #通过activation参数指定
# model.add(layers.Dense(10))
# model.add(layers.Activation(tf.nn.softmax))  # 显式添加layers.Activation激活层
# tf.print(model.summary())

# 自定义模型层
# mypower = layers.Lambda(lambda x:tf.math.pow(x,2))
# tf.print(mypower(tf.range(5)))

class Linear(layers.Layer):
    def __init__(self,units=32,**kwargs):
        super(Linear,self).__init__(**kwargs)
        self.units = units
    # build方法定义Layer需要被训练的参数
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear,self).build(input_shape)   # 相当于设置self.built=True
    # call方法一般定义正向传播运算逻辑，__call__方法调用了它
    def call(self, inputs, **kwargs):
        return tf.matmul(inputs,self.w) +self.b
    # 如要让自定义的Layer通过Function API组合成模型时可以序列化，需要自定义get_config方法
    def get_config(self):
        config = super(Linear,self).get_config()
        config.update({'units':self.units})
        return config

# linear = Linear(units=8)
# tf.print(linear.built)
# # 指定input_shape，显式调用build方法，第0维代表样本数量，用None填充
# linear.build(input_shape=(None,16))
# tf.print(linear.built)
# tf.print(linear.compute_output_shape(input_shape=(None,16)))


# 以下为Day8工作
# # 设置16个单元的线性层
# linear = Linear(units=16)
# # 如果built = False，调用__call__时会先调用build方法, 再调用call方法。
# tf.print(linear.built)
# linear(tf.random.uniform((100,64)))
# tf.print(linear.built)
# config=linear.get_config()
# tf.print(config)

# 查看模型中的参数情况
tf.keras.backend.clear_session()
model = models.Sequential()
model.add(Linear(units=16,input_shape=(64,)))
tf.print("model.input_shape:",model.input_shape)
tf.print("model.output_shape:",model.output_shape)
tf.print(model.summary())




tf.print("success")