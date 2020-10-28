# coding=utf-8
import tensorflow as tf
from tensorflow.python.framework import graph_util


def main(): 
    tf.reset_default_graph() 

    with tf.Session() as sess:
        # load the meta graph and weights
        saver = tf.train.import_meta_graph('./model/mnist_model-4000.meta')
        # get weights
        saver.restore(sess, tf.train.latest_checkpoint("./model/"))
        # 设置输出类型以及输出的接口名字
        graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["label"])
        tf.train.write_graph(graph, './frozen_pb_model', 'frozen_model_meta.pb', as_text = False) 
    print("done") 

if __name__ == '__main__': 
    main()
