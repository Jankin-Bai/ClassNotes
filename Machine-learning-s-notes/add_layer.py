import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np


########################################################################################
# 定义一个添加层函数,包含(参数,计算方式,激励函数activation_function)

def add_layer(inputs, in_size, out_size, activation_function=None):
    #定义一个[in_size,out_size]的随机变量矩阵in_size*out_size
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #定义一个[1,out_size]的为0矩阵再+0.1,全为0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    #定义计算方式output=Weights*input+biases
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #判断是否添加激励函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        #output=activation_function(Weights*input+biases)
        outputs = activation_function(Wx_plus_b)
    return outputs
########################################################################################

########################################################################################
# 构建所需数据 加入noise使之看起来更像真实情况
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

########################################################################################

#利用占位符tf.placeholder()定义 神经网络的输入,None代表无论输入多少都可以,1表示只有1个特征
xs = tf.placeholder(tf.float32,[None, 1])
ys = tf.placeholder(tf.float32,[None, 1])

########################################################################################
#搭建网络

# 定义神经层
# 包括: 输入层(input layer) 隐藏层(hide layer) 输出层(output layer)
#            1个                    10个           1个

# 定义隐藏层 激励函数采用nn.relu
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

# 定义输出层 输入为l1(hide layer))
prediction = add_layer(l1, 10, 1, activation_function=None)

########################################################################################

# 计算预测值prediction和真实值的误差,差的平方和取平均.
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

# 让机器学习提升它的准确率 tf.tran.GrandientDescentOptimizer(),0.1的效率
# GradientDescent(梯度下降) Optimizer(优化)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

########################################################################################

#变量初始化
init = tf.global_variables_initializer()
#定义Session,并用Sessino来执行初始化步骤
sess = tf.Session()
sess.run(init)
########################################################################################
#     TensorFlow中只有session.run()才会执行我们的定义
########################################################################################

# 训练

for i in range(1000):
    #让机器学习1000次,内容是train_step
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    

    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))

########################################################################################
