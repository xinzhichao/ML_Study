import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import time
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


def linear_function():
    """
        实现一个线性功能：
            初始化W，类型为tensor的随机变量，维度为(4,3)
            初始化X，类型为tensor的随机变量，维度为(3,1)
            初始化b，类型为tensor的随机变量，维度为(4,1)
        返回：
            result - 运行了session后的结果，运行的是Y = WX + b

        """

    np.random.seed(1)  # 指定随机种子

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul是矩阵乘法
    # Y = tf.matmul(W,X) + b #也可以以写成这样子

    # 创建一个session并运行它
    sess = tf.Session()
    result = sess.run(Y)

    # session使用完毕，关闭它
    sess.close()

    return result

def sigmoid(Z):
    x = tf.placeholder(tf.float32,name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict = {x:Z})
    return result


def one_hot_matrix(lables, C):
    """
    创建一个矩阵，其中第i行对应第i个类号，第j列对应第j个训练样本
    所以如果第j个样本对应着第i个标签，那么entry (i,j)将会是1
    参数：
        lables - 标签向量
        C - 分类数
    返回：
        one_hot - 独热矩阵
    """
    # 创建一个tf.constant，赋值为C，名字叫C
    C = tf.constant(C, name="C")
    # 使用tf.one_hot，注意一下axis
    one_hot_matrix = tf.one_hot(indices=lables, depth=C, axis=0)
    # 创建一个session
    sess = tf.Session()
    # 运行session
    one_hot = sess.run(one_hot_matrix)
    # 关闭session
    sess.close()
    return one_hot

labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels,C=4)
print(str(one_hot))