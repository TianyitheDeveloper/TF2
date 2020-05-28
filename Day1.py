import tensorflow as tf
# tf.keras.backend.clear_session()

# tf.print("TF版本:", tf.__version__)
# a = tf.constant("hello")
# b = tf.constant("TF2")
# c = tf.strings.join([a,b]," ")
# tf.print(c)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers
import seaborn as sns

dftrain_raw = pd.read_csv('data/titanic/train.csv')
dftest_raw = pd.read_csv('data/titanic/test.csv')
# print(dftrain_raw.head(10))

# print(dftrain_raw['Survived'].value_counts())                   # 查看空准确率
# sns.heatmap(dftrain_raw.corr())                                 # 使用热图查看各特征相关度
# plt.show()

def preprocessing(dfdata):
    """
    第一步：处理数据
    :param dfdata:
    :return:
    """
    dfresult = pd.DataFrame()                                           # 定义一个空表
    # print(dfresult)
    dfPclass = pd.get_dummies(dfdata['Pclass'])                         # 定类数据用多米
    dfPclass.columns = ['Pclass_'+str(x) for x in dfPclass.columns]     # 加一个小表头
    dfresult = pd.concat([dfresult,dfPclass],axis=1)                    # 把刚刚这个数据加到空表中去

    dfSex = pd.get_dummies(dfdata['Sex'])                               # 定类数据用多米
    dfresult = pd.concat([dfresult,dfSex],axis=1)                       # 把刚刚这个数据加到空表中去

    dfresult['Age'] = dfdata['Age'].fillna(0)                           # 也可直接在表中先写一个表头，然后把数据补上
    # dfAge = dfdata['Age'].fillna(0)                                     # 把年龄中的缺失值用0补齐
    # dfresult = pd.concat([dfresult,dfAge],axis=1)                       # 把刚刚这个数据加到空表中去
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')       # 加一个空值标记（astype可直接转换一列数据的格式）

    dfresult['SibSp'] = dfdata['SibSp']                                 # 直接在表中先写一个表头，然后把数据补上
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')   # 判空标记

    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)       # 定类数据用多米，同时把空值也当作一类
    dfEmbarked.columns = ['Embarked_'+str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis=1)

    return dfresult

# 处理数据
x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw['Survived'].values
x_test = preprocessing(dftest_raw).values
y_test = dftest_raw['Survived'].values
tf.print(x_train.shape,x_test.shape)

# 定义模型
tf.keras.backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(15,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# print(model.summary())

# 训练模型
model.compile(optimizer='adam',  # 优化器为adam
              loss='binary_crossentropy',  # 损失函数为二元交叉熵
              metrics=['AUC'])  # 评价标准为AUC
history = model.fit(x_train, y_train,  # 训练数据和标签
                        batch_size=64,  # 每一批为64个数据
                        epochs=30,  # 跑30次
                        validation_split=0.2)  # 分20%用于验证
# from tensorboard import notebook
# notebook.list()
# notebook.start("--logdir ./data/keras_model")
# 评估模型
train_metrics = history.history["AUC"]
val_metrics = history.history['val_' + "AUC"]
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics, 'bo--')
plt.plot(epochs, val_metrics, 'ro--')
plt.title('Training and validation' + "AUC")
plt.xlabel('Epochs')
plt.ylabel("AUC")
plt.legend(["train_" + "AUC", 'val_' + "AUC"])
plt.show()
train_metrics = history.history["loss"]
val_metrics = history.history['val_' + "loss"]
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics, 'bo--')
plt.plot(epochs, val_metrics, 'ro--')
plt.title('Training and validation' + "loss")
plt.xlabel('Epochs')
plt.ylabel("AUC")
plt.legend(["train_" + "loss", 'val_' + "loss"])
plt.show()
model.evaluate(x=x_test, y=y_test)

# 使用模型
model.predict(x_test[0:10])
model.predict_classes(x_test[0:10])

# 保存模型
model.save('data/first_model.h5')                           # 保存模型结构及权重
del model                                                   # 删除现有模型




