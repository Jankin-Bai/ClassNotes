import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#定义添加层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weight'):
          Weight = tf.Variable(tf.random_normal([in_size,out_size]),name = 'W')
        with tf.name_scope('biases'):
          biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name ='b')
        with tf.name_scope('Wx_plus_b'):

          Wx_plus_b = tf.matmul(inputs, Weight) + biases

          if activation_function is None:
              outputs = Wx_plus_b
          else:
              outputs = activation_function(Wx_plus_b)
          return outputs
#模拟输入数据（加入噪声）

#我们用 tf.Variable 来创建描述 y 的参数. 
# 我们可以把 y_data = x_data*0.1 + 0.3 想象成 y=Weights * x + biases, 
# 然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3.
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.005,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None, 1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None, 1],name = 'y_input')

#搭建网络

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]),name='loss')
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#将框架加载到文件中
writer = tf.summary.FileWriter("logs/",sess.graph)

#输出原始数据
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

# 训练

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs: x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)
