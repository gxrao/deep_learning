import tensorflow as tf

# prepare the data
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('..\\..\\MNIST\\', one_hot=True)

# create the data graph
sess = tf.InteractiveSession()
# add the name attribute for the mode application
x = tf.placeholder(tf.float32, shape=[None,784],name='x')
y_ = tf.placeholder(tf.float32,shape=[None,10])

# parameters setting
batch_size = 50
max_steps = 4000
learning_rate = 0.0001

# init weight and bias
def init_variable(w_shape,b_shape):
    weight = tf.truncated_normal(w_shape,stddev=0.1)
    bias = tf.constant(0.1,'float')
    return tf.Variable(weight),tf.Variable(bias)


# create conv2d
def conv2d(x,w,b,keep_prob):
    # conv
    conv_res =  tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') + b
    # activation function
    activation_res = tf.nn.relu(conv_res)
    # pooling
    pool_res = tf.nn.max_pool(activation_res,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return tf.nn.dropout(pool_res, keep_prob)

x_img = tf.reshape(x,shape=[-1,28,28,1])
# filter_size = 5x5 in_channel = 1 out_channels =32
w_conv1,b_conv1 = init_variable([5,5,1,32],[32])
h_layer1 = conv2d(x_img,w_conv1,b_conv1,1.0)
w_conv2,b_conv2 = init_variable([5,5,32,64],[64])
h_layer2 = conv2d(h_layer1,w_conv2,b_conv2,1.0)
h_layer2_reshape = tf.reshape(h_layer2,shape=[-1,7*7*64])

# full connection layer
w_full_layer,b_full_layer = init_variable([7*7*64,1000],[1000])
full_layer = tf.nn.relu(tf.matmul(h_layer2_reshape,w_full_layer)+b_full_layer)
full_layer_dropout = tf.nn.dropout(full_layer,1.0)
# output layer
w_output_layer,b_output_layer = init_variable([1000,10],[10])
y = tf.nn.softmax(tf.matmul(full_layer_dropout,w_output_layer)+b_output_layer)

# define loss and select optimizer

loss = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# train the model
init = tf.global_variables_initializer()
init.run()

# predict_vale ,add the name attribute for the mode application
predict_value = tf.argmax(y, 1, name='predict_value')


predict = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(predict,'float'))

for i in range(max_steps):
    x_batch,y_batch = mnist.train.next_batch(batch_size)
    
    if (i+1)%100 == 0:
        print('training steps: %d'%i)
        a = acc.eval(feed_dict={x:x_batch,y_:y_batch})
        print('training accurary is %f'%a)
    train_step.run(feed_dict={x:x_batch,y_:y_batch})


# because it raise error when i select all test data.I select 2000 images as testing data
acc_test = acc.eval(feed_dict={x:mnist.test.images[:2000,:],y_:mnist.test.labels[:2000,:]})
print('test accurary is:%f'%acc_test)

# save the model
saver = tf.train.Saver()
saver.save(sess,'./cnn/cnn_simple/model/mnist.ckpt')
sess.close()






