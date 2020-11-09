import os
from queue import Queue
from threading import Thread
import pandas as pd
import args
import tensorflow as tf
import tokenization
import modeling
import optimization
import collections

#输入样本的一个单例
class InputExample(object):
    def __init__(self,guid,text_a,text_b=None,label=None):
        #guid全局唯一标识
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

#定义一个输入特征单例
class InputFeatures(object):
    def __init__(self,input_ids,input_mask,segment_ids,label_id):
        self.input_ids = input_ids#每个输入字符
        self.input_mask = input_mask#mask位置id
        self.segment_ids = segment_ids#句子id
        self.label_id =label_id#类别

#针对分类问题定义数据处理基类
class DataProcessor(object):
    def get_train_examples(self,data_dir):
        raise NotImplementedError()
    def get_dev_examples(self,data_dir):
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()

#基于基础类创建数据处理类
class ClassProcessor(DataProcessor):
    #定义迭代器输入数据
    def get_sentence_examples(self,questions):
        for index,data in enumerate(questions):
            guid = 'test-%d' %index
            text_a = tokenization.convert_to_unicode(str(data))
            text_b = None
            label =str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    #获取类别标签函数
    def get_labels(self):
        return ['0','1','2']

#定义整个处理流程类
class BertClassify:
    #初始化相关参数包括数据处理方式，分词接口，max_len,batch_size,estimator结构等
    def __init__(self,batch_size=args.batch_size):
        self.mode = None
        self.max_seq_length = args.max_seq_len
        self.tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = None
        self.processor = ClassProcessor()
        tf.logging.set_verbosity(tf.logging.INFO)
    
    def set_mode(self,mode):
        self.mode = mode
        self.estimator = self.get_estimator()
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.predict_thread.start()

    #定义获取estimator实例的函数
    def get_estimator(self):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig

        bert_config = modeling.BertConfig.from_json_file(args.config_name)
        label_list = self.processor.get_labels()

        init_checkpoint = args.output_dir
        #创建model_fn对象,该对象中定义了整个模型结构框架
        model_fn = self.model_fn_builder(bert_config=bert_config,num_labels=len(label_list),init_checkpoint=init_checkpoint,use_one_hot_embeddings=False)
        config_dis = tf.estimator.RunConfig()
        return Estimator(model_fn=model_fn,config=config_dis,model_dir=args.output_dir,params={'batch_size':self.batch_size})

    #定义model_fn结构函数
    def model_fn_builder(self,bert_config,num_labels,init_checkpoint,use_one_hot_embeddings):
        def model_fn(features,labels,mode,params):
            from tensorflow.python.estimator.model_fn import EstimatorSpec
            input_ids = features["input_ids"]
            input_mask =features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            probabilities = self.create_model(bert_config,input_ids,input_mask,segment_ids,label_ids,num_labels,use_one_hot_embeddings)

            tvars = tf.trainable_variables()

            if init_checkpoint:
                (assignment_map, initialized_variable_names) \
                    = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            output_spec = EstimatorSpec(mode=mode,predictions=probabilities)
            return output_spec
        return model_fn

    #定义模型结构函数
    def create_model(self,bert_config,input_ids,input_mask,segment_ids,labels,num_labels,use_one_hot_embeddings):
        #加载bert模型基础结构
        model = modeling.BertModel(config=bert_config,is_training=False,input_ids=input_ids,input_mask=input_mask,token_type_ids=segment_ids,use_one_hot_embeddings=use_one_hot_embeddings)
        #获取整个句子的输出
        output_layer = model.get_pooled_output()
        #以句子为单位获取输出层，然后摘取最后一层的参数信息
        hidden_size =output_layer.shape[-1].value
        #从新定义权重矩阵w,和b,根据最后一层的参数在结合新的w,b对模型进行微调
        output_weights = tf.get_variable("output_weights",[num_labels,hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable("output_bias",[num_labels],initializer=tf.zeros_initializer())

        #从新定义损失函数
        with tf.variable_scope("loss"):
            #微调新模型的输出
            logits = tf.matmul(output_layer,output_weights,transpose_b=True)
            logits = tf.nn.bias_add(logits,output_bias)
            probabilities = tf.nn.softmax(logits,axis=-1)
            log_probs = tf.nn.log_softmax(logits,axis=-1)
            one_hot_labels = tf.one_hot(labels,depth=num_labels,dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels*log_probs,axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (probabilities)

    #针对迭代器从新定义example转feature函数
    def convert_examples_to_features(self,examples,label_list,max_seq_length,tokenizer):
        for (ex_index,example) in enumerate(examples):
            label_map={}
            for (i,label)in enumerate(label_list):
                    label_map[label]=i
            tokens_a = tokenizer.tokenize(example.text_a)
            tf.logging.info("len tokens_a:%d",len(tokens_a))
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

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            label_id = label_map[example.label]
            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id)

            yield feature


    #输入生成器
    def generate_from_queue(self):
        while True:
            predict_examples = self.processor.get_sentence_examples(self.input_queue.get())
            features = list(self.convert_examples_to_features(predict_examples,self.processor.get_labels(),args.max_seq_len,self.tokenizer))
            yield {'input_ids':[f.input_ids for f in features],'input_mask':[f.input_mask for f in features],'segment_ids':[f.segment_ids for f in features],'label_ids':[f.label_id for f in features]}

    #预测函数
    def predict_from_queue(self):
        for i in self.estimator.predict(input_fn=self.queue_predict_input_fn,yield_single_examples=False):
            self.output_queue.put(i)
   
    #定义带有迭代器功能的输入
    def queue_predict_input_fn(self):
        return (tf.data.Dataset.from_generator(self.generate_from_queue,output_types={'input_ids': tf.int32,'input_mask': tf.int32,'segment_ids': tf.int32,'label_ids': tf.int32},output_shapes={'input_ids': (None, self.max_seq_length),'input_mask': (None, self.max_seq_length),'segment_ids': (None, self.max_seq_length),'label_ids': (1,)}).prefetch(10))

if __name__ == '__main__':
    cls = BertClassify()
    cls.set_mode(tf.estimator.ModeKeys.PREDICT)
    
    while True:
        inputsentence = input('输入句子:')
        cls.input_queue.put([inputsentence])
        predict = cls.output_queue.get()[0]
        print (predict.tolist().index(max(predict)))
    
