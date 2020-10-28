# coding=utf-8
import tensorflow as tf
from tensorflow.python.tools import freeze_graph 

# network是你自己定义的模型
from mnist_train import forward

# 模型的checkpoint文件地址
ckpt_path = "./model/mnist_model-4000"

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
			input_graph='./model/mnist_model.pbtxt',
			input_saver='',
			input_binary=False, 
			input_checkpoint=ckpt_path, 
			output_node_names='label',
			restore_op_name='save/restore_all',
			filename_tensor_name='save/Const:0',
			output_graph='./frozen_pb_model/frozen_model_pbtxt.pb',
			clear_devices=False,
			initializer_nodes='',
		)	

	print("done") 

if __name__ == '__main__': 
	main()
