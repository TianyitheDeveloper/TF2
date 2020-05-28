import tensorflow as tf
import numpy as np
from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# iris = datasets.load_iris()
# ds1 = tf.data.Dataset.from_tensor_slices((iris["data"],iris["target"]))
# for features, label in ds1.take(5):
#     tf.print(features,label)

# dfiris = pd.DataFrame(iris["data"],columns=iris.feature_names)
# ds2 = tf.data.Dataset.from_tensor_slices((dfiris.to_dict("list"),iris["target"]))
# for features,label in ds2.take(5):
#     tf.print(features,label)

# 从Python generator构建数据管道(缺失)

# ds4 = tf.data.experimental.make_csv_dataset(
#     file_pattern=["data/titanic/train.csv","data/titanic/test.csv"],
#     batch_size=3,
#     label_name="Survived",
#     na_value="",
#     num_epochs=1,
#     ignore_errors=True
# )
# for data,label in ds4.take(5):
#     tf.print(data,label)

# ds5 = tf.data.TextLineDataset(
#     filenames=["data/titanic/train.csv","data/titanic/test.csv"]
# ).skip(1)
# for line in ds5.take(5):
#     tf.print(line)

# ds6 = tf.data.Dataset.list_files("data/cifar2/train/*/*.jpg")
# for i in ds6.take(5):
#     tf.print(i)
# def load_image(img_path,size=(32,32)):
#     label = 1 if tf.strings.regex_full_match(img_path,".*/automobile/.*") else 0
#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_jpeg(img)
#     img = tf.image.resize(img,size)
#     return (img,label)
# for i,(img,label) in enumerate(ds6.map(load_image).take(2)):
#     plt.figure(i)
#     plt.imshow((img / 255.0).numpy())
#     plt.title("label = %d" % label)
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# 从tfrecords文件构建数据管道(缺失)

# map:将转换函数映射到数据集每一个元素
# ds = tf.data.Dataset.from_tensor_slices(["hello world","hello TF2","hello CN"])
# ds_map = ds.map(lambda x:tf.strings.split(x," "))
# for x in ds_map:
#     tf.print(x)

# flat_map:将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平
# ds = tf.data.Dataset.from_tensor_slices(["hello world","hello TF2","hello CN"])
# ds_flatmap = ds.flat_map(lambda x:tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))
# for x in ds_flatmap:
#     tf.print(x)

# interleave:效果类似flat_map,但可以将不同来源的数据夹在一起。
# ds = tf.data.Dataset.from_tensor_slices(["hello world","hello TF2","hello CN"])
# ds_interleave = ds.interleave(lambda x:tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))
# for x in ds_interleave:
#     tf.print(x)

# filter:过滤掉某些元素。
# ds = tf.data.Dataset.from_tensor_slices(["hello world","hello TF2","hello CN"])
# ds_filter = ds.filter(lambda x:tf.strings.regex_full_match(x,".*[C|B].*"))
# for x in ds_filter:
#     tf.print(x)

# zip:将两个长度相同的Dataset横向铰合
# ds1 = tf.data.Dataset.range(0,3)
# ds2 = tf.data.Dataset.range(3,6)
# ds3 = tf.data.Dataset.range(6,9)
# ds_zip = tf.data.Dataset.zip((ds1,ds2,ds3))
# for x,y,z in ds_zip:
#     tf.print(x.numpy(),y.numpy(),z.numpy())

# condatenate:将两个Dataset纵向连接。
# ds_concat = tf.data.Dataset.concatenate(ds1,ds2)
# for i in ds_concat:
#     tf.print(i)

# reduce:执行归并操作
# ds = tf.data.Dataset.from_tensor_slices([1,2,3,4,5.6])
# result = ds.reduce(1.0,lambda x,y:tf.add(x,y))
# tf.print(result)

# batch:构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch
# ds = tf.data.Dataset.range(12)
# ds_batch = ds.batch(4)
# for i in ds_batch:
#     tf.print(i)

# padded_batch:构建批次，类似batch, 但可以填充到相同的形状
# elements = [[1,2],[3,4,5],[6,7],[8]]
# ds = tf.data.Dataset.from_generator(lambda :iter(elements),tf.int32)
# ds_padded_batch = ds.padded_batch(3,padded_shapes=[4,],padding_values=-1)
# for i in ds_padded_batch:
#     tf.print(i)

# window:构建滑动窗口，返回Dataset of Dataset
# ds = tf.data.Dataset.range(12)
# ds_window = ds.window(3,shift=1).flat_map(lambda x:x.batch(3,drop_remainder=True))
# for x in ds_window:
#     tf.print(x)

# shuffle:数据顺序洗牌
# ds = tf.data.Dataset.range(12)
# ds_shuffle = ds.shuffle(buffer_size=5)
# for x in ds_shuffle:
#     tf.print(x)

# repeat:重复数据若干次，不带参数时，重复无数次
# ds = tf.data.Dataset.range(3)
# ds_repeat = ds.repeat(3)
# for i in ds_repeat:
#     tf.print(i)

# shard:采样，从某个位置开始隔固定距离采样一个元素
# ds = tf.data.Dataset.range(12)
# ds_shard = ds.shard(3,index=2)
# for i in ds_shard:
#     tf.print(i)

# take:采样，从开始位置取前几个元素
# ds = tf.data.Dataset.range(12)
# ds_take = ds.take(3)
# tf.print(list(ds_take.as_numpy_iterator()))

tf.print("success")