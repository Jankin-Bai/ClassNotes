import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt #可视化模块

#terminal type tensorboard --logdir='logs/' to make graph
#cleck the http to open.
###############################################################################
#定义添加层函数(实质上是神经元)
def add_layer(inputs,
    in_size,out_size,
    n_layer,
    activation_function=None):
    layer_name = 'layer%s'%n_layer    #标记定义当前层
    with tf.name_scope('layer_name'):   #将当前层函数显示在一个大的框架里
        with tf.name_scope('Weights'):
          Weights = tf.Variable(tf.random_normal([in_size,out_size]),
          name = 'W')
          #tf.summary.histogram(图表名称，图表要记录的变量)用来显示图表
          tf.summary.histogram(layer_name +'/Weights',Weights)
        with tf.name_scope('biases'):
          biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,
          name ='b')
          tf.summary.histogram(layer_name +'/biases',biases)
        with tf.name_scope('Wx_plus_b'):

          Wx_plus_b = tf.matmul(inputs, Weights) + biases

          if activation_function is None:
              outputs = Wx_plus_b
          else:
              outputs = activation_function(Wx_plus_b)
              tf.summary.histogram(layer_name +'/output',outputs)
          return outputs

###############################################################################

#产生数据
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.005,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#占位符申请变量空间

#使用with tf.name_scope('inputs')可以将xs、ys包含进来，形成一个大的图层，图层的名字就是inputs
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None, 1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None, 1],name = 'y_input')

###############################################################################

#搭建网络

l1 = add_layer(xs,1,10,n_layer = 1,activation_function=tf.nn.relu)

l2 = add_layer(l1,10,10,n_layer = 2,activation_function=tf.nn.relu)

prediction = add_layer(l2, 10, 1,n_layer = 3, activation_function=None)

#loss区用来看神经网络训练是有效果的.
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]),
    name='loss')
#观看loss的变化比较重要. 当你的loss呈下降的趋势,说明你的神经网络训练是有效果的.
tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

###############################################################################
sess = tf.Session()

#打包 tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起
merged = tf.summary.merge_all() 
#将框架加载到文件中
writer = tf.summary.FileWriter("logs/",sess.graph)

###############################################################################

sess.run(tf.global_variables_initializer())

#构建图形，用散点图描述真实数据之间的关系
fig = plt.figure()#生成图画框
ax = fig.add_subplot(1,1,1) #
ax.scatter(x_data,y_data)   #用点的形式将真实数据打印出来
plt.ioff()   #用于连续显示

###############################################################################

# 训练

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        #在图表中打印出数据
        rs = sess.run(merged,feed_dict = {xs:x_data,ys:y_data})
        writer.add_summary(rs,i)
        #
        
        #先抹除再打印，ax.lines第一次并未定义
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs: x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)#把训练值用红色宽度为5的线打印出来
        plt.pause(0.1) #暂停0.1秒
        print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))

###############################################################################
plt.show()  #只show 1次
