# coding=utf-8
from tensorflow.python.tools import freeze_graph 
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf

# network是你自己定义的模型
from mnist_train import forward

def main(): 
	tf.reset_default_graph() 

	x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')

	# flow是模型的输出
	pred = forward(x)
	#设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用 
	pred = tf.cast(pred, tf.int8, 'label')

	with tf.Session() as sess:
		# 把图和参数结构一起，注意input_binary 
		freeze_graph.freeze_graph(
			input_graph=None,
			input_saver='',
			input_binary=False, 
			input_checkpoint=None, 
			output_node_names='label',
			restore_op_name='save/restore_all',
			filename_tensor_name='save/Const:0',
			output_graph='./frozen_pb_model/frozen_model_saved_model.pb',
			clear_devices=False,
			initializer_nodes='',
            input_saved_model_dir="./saved_model",
            saved_model_tags= tag_constants.SERVING
		)	

	print("done") 

if __name__ == '__main__': 
	main()
