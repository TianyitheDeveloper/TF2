import tensorflow as tf
from tensorflow.keras import models,layers,optimizers
import pandas as pd

data = pd.read_excel('train.xlsx')

X = tf.constant(data.drop('y',axis=1).values,dtype=tf.float32,shape=[2000,4])
Y = tf.constant(data['y'],dtype=tf.float32,shape=[2000,1])

linear = models.Sequential()
linear.add(layers.Dense(1,input_shape=(4,)))

linear.compile(optimizer=optimizers.SGD(0.001), loss="mse",metrics=['mse'])
linear.fit(X,Y,batch_size=200,epochs=200)

tf.print("w=",linear.layers[0].kernel)
tf.print("b=",linear.layers[0].bias)
