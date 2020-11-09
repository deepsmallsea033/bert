import os
from queue import Queue
from threading import Thread
import args
import tensorflow as tf
import tokenization
import modeling
import optimization
import collections
import numpy as np
#输入样本的一个单例
class InputExample(object):
    def __init__(self,guid,text_a,text_b=None,label=None):
        #guid全局唯一标识
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

#定义一个输入特征单例
class InputFeatures(object):
    def __init__(self,input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids):
        self.input_ids = input_ids#每个输入字符
        self.input_mask = input_mask#mask位置id
        self.segment_ids = segment_ids#句子id
        self.masked_lm_positions = masked_lm_positions#mask位置
        self.masked_lm_ids = masked_lm_ids


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

#基于基础类创建数据处理类
class ClassProcessor(DataProcessor):
    #定义迭代器输入数据
    def get_sentence_examples(self,questions):
        for index,data in enumerate(questions):
            guid = 'test-%d' %index
            text_a = tokenization.convert_to_unicode(str(data))
            text_b = None
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b)


#定义整个处理流程类
class rnn_ppl_score:
    #初始化相关参数包括数据处理方式，分词接口，max_len,batch_size,estimator结构等
    def __init__(self,batch_size=args.batch_size):
        self.mode = None
        self.all_tokens = None
        self.max_seq_length = args.max_seq_len
        self.tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = None
        self.processor = ClassProcessor()
        tf.logging.set_verbosity(tf.logging.INFO)
        self.MASKED_TOKEN = "[MASK]"
        self.MASKED_ID = self.tokenizer.convert_tokens_to_ids([self.MASKED_TOKEN])[0]
    
    def set_mode(self,mode):
        self.mode = mode
        self.estimator = self.get_estimator()
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.input_queue = Queue(maxsize=1)
            self.output_queue = Queue(maxsize=1)
            self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
            self.predict_thread.start()

    def gather_indexes(self,sequence_tensor, positions):
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]#bert模型中batch_size
        seq_length = sequence_shape[1]#bert模型中最大序列长度
        width = sequence_shape[2]#单词的维度
        #flat_offsets 首先定义一个0-batch_size行矩阵，然后矩阵中每个值乘以seq_length，然后变量列为1的矩阵
        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,[batch_size * seq_length, width])
        #根据索引在flat_sequence_tensor中抽取对应mask的词输出
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor,batch_size,seq_length
    
    #定义新模型中
    def get_masked_lm_output(self,bert_config,input_tensor, output_weights, positions,label_ids):
        input_tensor,size,max_len = self.gather_indexes(input_tensor,positions)
        with tf.variable_scope("cls/predictions"):
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(input_tensor,units=bert_config.hidden_size,activation=modeling.get_activation(bert_config.hidden_act),kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)
            output_bias = tf.get_variable("output_bias",shape=[bert_config.vocab_size],initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            label_ids = tf.reshape(label_ids, [-1])

            one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            loss = tf.reshape(per_example_loss, [-1, tf.shape(positions)[1]])
        return loss
    
    def get_estimator(self):

        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig

        bert_config = modeling.BertConfig.from_json_file(args.config_name)

        init_checkpoint = args.ckpt_name
        model_fn = self.model_fn_builder(
                bert_config=bert_config,
                init_checkpoint=init_checkpoint,
                use_one_hot_embeddings=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
        config.log_device_placement = False

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config), model_dir=args.model_dir,
                         params={'batch_size': self.batch_size})

    #定义模型结构函数
    def create_model(self,bert_config,input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,use_one_hot_embeddings):
        #从新定义模型结构在原bert基础模型基础上增加新的变量参数
        model = modeling.BertModel(config=bert_config,is_training=False,input_ids=input_ids,input_mask=input_mask,token_type_ids=segment_ids,use_one_hot_embeddings=use_one_hot_embeddings)
        #model.get_sequence_output()获取token级的模型output[batch_size, seq_length, embedding_size]
        masked_lm_example_loss = self.get_masked_lm_output(bert_config, model.get_sequence_output(), model.get_embedding_table(),masked_lm_positions, masked_lm_ids)

        return masked_lm_example_loss

    #定义model_fn结构函数
    def model_fn_builder(self,bert_config,init_checkpoint,use_one_hot_embeddings):
        def model_fn(features,mode,params):
            from tensorflow.python.estimator.model_fn import EstimatorSpec

            input_ids = features["input_ids"]
            input_mask =features["input_mask"]
            segment_ids = features["segment_ids"]
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_ids"]

            masked_lm_example_loss = self.create_model(bert_config,input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,use_one_hot_embeddings)
          
            tvars = tf.trainable_variables()
            initialized_variable_names = {}

            if init_checkpoint:
                (assignment_map, initialized_variable_names) \
                    = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            output_spec = EstimatorSpec(mode=mode,predictions=masked_lm_example_loss) 
            return output_spec 
        return model_fn

    def is_subtoken(self,x):
          return x.startswith("##")

    def create_masked_lm_prediction(self,input_ids, mask_position, mask_count=1):
        new_input_ids = list(input_ids)
        masked_lm_labels =[]
        masked_lm_positions = list(range(mask_position, mask_position + mask_count))
        for i in masked_lm_positions:
            new_input_ids[i] = self.MASKED_ID
            masked_lm_labels.append(input_ids[i])
        return new_input_ids, masked_lm_positions, masked_lm_labels

    #针对迭代器从新定义example转feature函数
    def convert_examples_to_features(self,examples,max_seq_length,tokenizer):
        for (ex_index,example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b =None
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            self.all_tokens = tokens
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
           
            i = 1
            while i<len(tokens)-1:
                mask_count = 1
                while self.is_subtoken(tokens[i+mask_count]):
                    mask_count +=1
                input_ids_new,masked_lm_positions,masked_lm_labels = self.create_masked_lm_prediction(input_ids, i, mask_count)
                while len(masked_lm_positions) < max_seq_length:
                    masked_lm_positions.append(0)
                    masked_lm_labels.append(0)

                feature = InputFeatures(input_ids=input_ids_new,input_mask=input_mask,segment_ids=segment_ids,masked_lm_positions=masked_lm_positions,masked_lm_ids=masked_lm_labels)
                i += mask_count
                
                yield feature


    #输入生成器
    def generate_from_queue(self):
        while True:
            predict_examples = self.processor.get_sentence_examples(self.input_queue.get())
            features = list(self.convert_examples_to_features(predict_examples,args.max_seq_len,self.tokenizer))

            yield {'input_ids':[f.input_ids for f in features],'input_mask':[f.input_mask for f in features],'segment_ids':[f.segment_ids for f in features],'masked_lm_positions':[f.masked_lm_positions for f in features],'masked_lm_ids':[f.masked_lm_ids for f in features]}

    #预测函数
    def predict_from_queue(self):
        for i in self.estimator.predict(input_fn=self.queue_predict_input_fn,yield_single_examples=False):
            self.output_queue.put(i)
   
    #定义带有迭代器功能的输入通过生成器函数生成dataset数据，生成dataset的目的是为了进一步生成estimator对象
    def queue_predict_input_fn(self):
        return (tf.data.Dataset.from_generator(self.generate_from_queue,output_types={'input_ids': tf.int32,'input_mask': tf.int32,'segment_ids': tf.int32,'masked_lm_positions': tf.int32,'masked_lm_ids':tf.int32},output_shapes={'input_ids': (None, self.max_seq_length),'input_mask': (None, self.max_seq_length),'segment_ids': (None, self.max_seq_length),'masked_lm_positions': (None,self.max_seq_length),'masked_lm_ids':(None,self.max_seq_length)}).prefetch(10))

#根据loss获取整个句子的得分
def get_content_score(result,content):
    length = len(content)
    content_loss = 0.0
    for i in range(length):
        content_loss +=float(result[i][0])

    score = float(np.exp(content_loss/length))
    return score

if __name__ == '__main__':
    cls = rnn_ppl_score()
    cls.set_mode(tf.estimator.ModeKeys.PREDICT)
    
    while True:
        inputsentence = input('输入句子:')
        cls.input_queue.put([inputsentence])

        predict = cls.output_queue.get()
        tokens = cls.all_tokens
        score = get_content_score(predict,inputsentence)
        print ("输入的句子：%s,句子得分：%f" %(inputsentence,score))
    
