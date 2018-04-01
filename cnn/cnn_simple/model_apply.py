import tensorflow as tf 

# prepare data
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('../../MNIST/',one_hot=True)

saver = tf.train.import_meta_graph(r'./cnn/cnn_simple/model/mnist.ckpt.meta') 
sess = tf.InteractiveSession()
saver.restore(sess, r'./cnn/cnn_simple/model/mnist.ckpt')  
graph = tf.get_default_graph()
x = graph.get_tensor_by_name('x:0')

predict_value = graph.get_tensor_by_name('predict_value:0')
p = sess.run(predict_value,feed_dict={x:mnist.test.images[:100,:]})
print(p)
