import tensorflow as tf
from google.protobuf import text_format

def read_pb():
    # as_text=False
    with tf.Session() as sess:
        with open('./model/mnist_model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
            tensor_name = [tensor.name for tensor in graph_def.node]
            print(tensor_name)
            
            tf.import_graph_def(graph_def,name="")
            graph = tf.get_default_graph()
            pred = graph.get_tensor_by_name("output:0")
            print(pred)

def read_pbtxt():
    # as_text=True
    with tf.Session() as sess:
        # 不使用'rb'模式
        with open('./model/mnist_model.pbtxt', 'r') as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)
            
            tensor_name = [tensor.name for tensor in graph_def.node]
            print(tensor_name)

            tf.import_graph_def(graph_def,name="")
            graph = tf.get_default_graph()
            pred = graph.get_tensor_by_name("output:0")
            print(pred)

if __name__ == "__main__":
    #read_pb()
    read_pbtxt()
