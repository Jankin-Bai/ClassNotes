#利用TensorFlow
#名词解释:
#        weights:权重
#        biases:偏量
#        random:随机变量
#        loss:误差
#        optimizer:优化
#        train:训练
#        initialize:初始化
#        range:范围
#        Gradient Descent:梯度下降
#        Session:会话

import tensorflow as tf
import numpy as np

#创建数据(creat data)
 x_data = np.random.rand(100).astype(np.float32)
 y_data = x_data*0.1+0.3

#以下structure建立了神经网络的结构,并初始化变量
###create tensorflow structure start ###
 #创建模型
  Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
  biases = tf.Variable(tf.zeros([1]))
  y = Weights*x_data+biases

 #计算误差(y和y_data的误差)
  loss = tf.reduce_mean(tf.square(y-y_data))

 #传播误差
  optimizer = tf.train.GradientDescentOptimizer(0.5)
  train = optimizer.minimize(loss) #更新

 #初始化
  init = tf.global_variables_initializer()
###create tensorflow structure end ###

#创建会话
 sess = tf.Session()
 sess.run(init)   #Very important 激活初始化,类似指针指向init

#开始训练
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases)) 
        