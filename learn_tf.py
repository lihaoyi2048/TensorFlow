# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/05/31 13:59

import tensorflow as tf


# 加载mnist数据
# mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。
# 同时提供了一个函数，用于在迭代中获得minibatch
import input_data
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

# 构建Softmax回归模型
# 建立一个拥有一个线性层的siftmax回归模型

# 为输入图像和目标输出类别创建节点，开始构建计算图
# 输入图片x是一个2维的浮点型张量，784是一张平展的MNIST图片的维度，None表示其值大小不定，
# 在这里作为第一个维度，用以指代batch的大小，即x的数量不定。
x = tf.placeholder("float", [None, 784])

# 定义权重W，初始化为0向量，784*10的矩阵。
W = tf.Variable(tf.zeros([784, 10]))
# 定义偏置b，初始化为0向量，是一个10维的向量。
b = tf.Variable(tf.zeros([10]))

# 回归模型
# 把向量化的图片x和权重矩阵W相乘，加上偏置b，然后计算每个分类的siftmax概率值
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 输出类别y_也是一个2维张量，其中每一行为一个10维的one-hot向量，用于表示对应某一MNIST图片的类别
y_ = tf.placeholder("float", [None, 10])

# 为训练过程指定最小化误差用的损失函数
# 模型的损失函数，是目标类别和预测类别之间的交叉熵，
# tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来，
# 交叉熵是指整个minibatch的
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 变量需要通过seesion初始化后，才能在session中使用
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
# sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(init)

# 使用最快下降法让交叉熵下降，步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 每一步迭代，加载100个样本，执行一次train_step，
# 并通过feed_dict将x和y_张量占位符用训练数据替代
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 模型评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 计算在测试数据上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

