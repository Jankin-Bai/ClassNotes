####################################################################################
###########################TensorFlow语法集锦########################################

# Session 对话

print("Session测试结果")

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

#matrix mlutiply(矩阵乘法,内积) np.dot(m1,m2)
product = tf.matmul(matrix1,matrix2)  

#因为 product 不是直接计算的步骤,
#所以我们会要使用 Session 来激活 product 并得到计算结果.
#有两种形式使用会话控制 Session 

#method 1
#sess = tf.Session()
#result = sess.run(product)
#
#print(result)
#sess.close()


#method 2
#有一些任务，可能事先需要设置，事后做清理工作。
#对于这种场景，Python的with语句提供了一种非常方便的处理方式。
#一个很好的例子是文件处理，你需要获取一个文件句柄，
#从文件中读取数据，然后关闭文件句柄。
#除了有更优雅的语法，with还可以很好的处理上下文环境产生的异常。下面是with版本的代码：

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
#这看起来充满魔法，但不仅仅是魔法，Python对with的处理还很聪明。
# 基本思想是with所求值的对象必须有一个__enter__()方法，一个__exit__()方法。
#紧跟with后面的语句被求值后，返回对象的__enter__()方法被调用，这个方法的返回值将被赋值给as后面的变量。
#当with后面的代码块全部被执行完之后，将调用前面返回对象的__exit__()方法。


####################################################################################

# Variable 变量定义
print("Variable测试结果")


state = tf.Variable(0,name='counter')

#定义常量 one
one = tf.constant(1)

#定义加法步骤(此步并未直接计算))
new_value = tf.add(state,one)

#将State更新成new_value
update = tf.assign(state,new_value)

#初始化变量<很重要>must have if define variable
init = tf.global_variables_initializer() 

#变量激活
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        #将sess的指针指向state才能print
        print(sess.run(state))

####################################################################################

# plecehoder 传入值
# placeholder 是 Tensorflow 中的占位符，暂时储存变量.

print("plecehoder测试结果")

#给定数据类型大小及结构
#TensorFlow中需要定义placeholder的type,一般为float32形式
input1 = tf.placeholder(tf.float32) 
input2 = tf.placeholder(tf.float32)

# multiply 是input1和input2做乘法运算,并输出为output
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #传值的工作交给了 sess.run() , 
    #需要传入的值放在了feed_dict={} 
    #并一一对应每一个 input.
    # placeholder 与 feed_dict={} 是绑定在一起出现的
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))