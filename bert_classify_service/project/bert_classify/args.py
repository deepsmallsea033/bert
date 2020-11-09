import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.dirname(__file__)

model_dir = os.path.join(file_path)
config_name = os.path.join(model_dir, 'chinese_L-12_H-768_A-12/bert_config.json')
ckpt_name = os.path.join(model_dir, 'chinese_L-12_H-768_A-12/bert_model.ckpt')
output_dir = os.path.join('/home/project/bert_classify/tmp/result/')
vocab_file = os.path.join(model_dir, 'chinese_L-12_H-768_A-12/vocab.txt')
data_dir = os.path.join('/home/project/bert_classify/data/')

num_train_epochs = 15
batch_size = 18
learning_rate = 0.00005

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# gpu使用率
gpu_memory_fraction = 0.9

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 180

# graph名字
graph_file = 'tmp/result/graph'
