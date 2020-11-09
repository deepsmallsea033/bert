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
from tensorflow.contrib import crf
import codecs
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
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
    @classmethod
    def _read_data(cls,input_file):
        with codecs.open(input_file,'r',encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                info = line.strip()
                tokens = info.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(info) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0 ])
                        lines.append([w,l])
                        words = []
                        labels = []
            return lines
                    
#基于基础类创建数据处理类
class ClassProcessor(DataProcessor):
    #获取训练数据函数统一成InputExample类型
    def get_train_examples(self,data_dir):
        examples = []
        lines = self._read_data(os.path.join(data_dir,"train1.txt"))
        for (i,line) in enumerate(lines):
            guid = 'train-%s' %i
            text_a = tokenization.convert_to_unicode(str(line[0]))
            text_b = None
            label = tokenization.convert_to_unicode(str(line[1]))
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples
    
    def get_test_examples(self,data_dir):
        lines = self._read_data(os.path.join(data_dir,"test.txt"))
        for (i,line) in enumerate(lines):
            guid = 'test-%s' %i
            text_a = tokenization.convert_to_unicode(str(line[0]))
            text_b = None
            label = tokenization.convert_to_unicode(str(line[1]))            
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

    #定义迭代器输入数据
    def get_sentence_examples(self,questions):
        for index,data in enumerate(questions):
            guid = 'test-%d' %index
            text_a = tokenization.convert_to_unicode(str(data))
            text_b = None
            label = ' '.join(['O' for _ in range(len(data))])
            tf.logging.info("input text:%s" %text_a)
            tf.logging.info("input label:%s" %label)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    #获取类别标签函数
    def get_labels(self):
        return ["O", "B-NAME", "I-NAME", "B-ORC", "I-ORC","B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

#定义整个处理流程类
class BertNer:
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
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.input_queue = Queue(maxsize=1)
            self.output_queue = Queue(maxsize=1)
            self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
            self.predict_thread.start()

    #定义模型结构函数
    def create_model(self,bert_config,is_training,input_ids,input_mask,segment_ids,labels,num_labels,use_one_hot_embeddings):
        #加载bert模型基础结构
        model = modeling.BertModel(config=bert_config,is_training=is_training,input_ids=input_ids,input_mask=input_mask,token_type_ids=segment_ids,use_one_hot_embeddings=use_one_hot_embeddings)
        #获取整个句子的输出
        output_layer = model.get_sequence_output()
        #以句子为单位获取输出层，然后摘取最后一层的参数信息
        hidden_size =output_layer.shape[-1].value
        #从新定义权重矩阵w,和b,根据最后一层的参数在结合新的w,b对模型进行微调
        output_weights = tf.get_variable("output_weights",[hidden_size,num_labels],initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable("output_bias",[num_labels],initializer=tf.zeros_initializer())
        lengths = tf.reduce_sum(tf.sign(tf.abs(input_ids)),reduction_indices=1)
        #从新定义损失函数
        with tf.variable_scope("logits"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer,keep_prob=0.9)
            output_layer = tf.reshape(output_layer,shape=[-1,hidden_size])
            #微调新模型的输出
            pred = tf.tanh(tf.nn.xw_plus_b(output_layer,output_weights,output_bias))

            logits =  tf.reshape(pred,[-1,args.max_seq_len,num_labels])
        
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable("transitions",[num_labels,num_labels],initializer=tf.truncated_normal_initializer())
            if labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                                    inputs=logits,
                                    tag_indices=labels,
                                    transition_params=trans,
                                    sequence_lengths=lengths)
                loss,trans =  tf.reduce_mean(-log_likelihood), trans
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=lengths)
        return (loss, logits, trans, pred_ids)
    #定义model_fn结构函数
    def model_fn_builder(self,bert_config,num_labels,init_checkpoint,learning_rate,num_train_steps,num_warmup_steps,use_one_hot_embeddings):
        def model_fn(features,labels,mode,params):
            from tensorflow.python.estimator.model_fn import EstimatorSpec
            input_ids = features["input_ids"]
            input_mask =features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            total_loss, logits, trans, pred_ids = self.create_model(bert_config,is_training,input_ids,input_mask,segment_ids,label_ids,num_labels,use_one_hot_embeddings)
          
            tvars = tf.trainable_variables()
            initialized_variable_names = {}

            if init_checkpoint:
                (assignment_map, initialized_variable_names) \
                    = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            if mode == tf.estimator.ModeKeys.TRAIN:
                #定义目标函数优化器
                train_op = optimization.create_optimizer(total_loss,learning_rate,num_train_steps,num_warmup_steps,False)             
                output_spec = EstimatorSpec(mode=mode,loss=total_loss,train_op=train_op)        
            elif mode == tf.estimator.ModeKeys.EVAL:
                #验证集需要调整训练过程中的一下参数比如准确度
                def metric_fn(per_example_loss,label_ids,logits):
                    pass
            else:
                output_spec = EstimatorSpec(mode=mode,predictions=pred_ids) 
            return output_spec 
        return model_fn

    #根据输入的inputexample拆分出inputfeature
    def convert_single_example(self,ex_index,example,label_list,max_seq_length,tokenizer):
        label_map = {}
        for (i,label)in enumerate(label_list):
            tf.logging.info("write label dict %s-%s" %(i,label))
            label_map[label] = i

        if not os.path.exists(os.path.join(args.output_dir, 'label2id.pkl')):
            with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'wb') as w:
                pickle.dump(label_map, w)

        textlist = example.text_a.split(' ')
        labellist = example.label.split(' ')
        tf.logging.info("textlist: %s" %example.text_a)
        tf.logging.info("labellist: %s" %example.label)
        tokens = []
        labels = []
        for i,word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m==0:
                    labels.append(label_1)
                else:
                    labels.append("X")

        if len(tokens)>max_seq_length -1:
            tokens = tokens[0:(max_seq_length -2)]
            labels = labels[0:(max_seq_length -2)]

        ntokens =[]
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])

        for i,token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1]*len(input_ids)
        while len(input_ids) <max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("**NULL**")
        tf.logging.info("input_ids %s" % (len(input_ids)))
        tf.logging.info("max_seq_length %s" % max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("guid: %s" % (example.guid))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in ntokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
 
        feature = InputFeatures(input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids,label_id=label_ids)
        return feature


    #将InputExample格式转换成tfrecord格式
    def file_based_convert_examples_to_features(self,examples,label_list,max_seq_length,tokenizer,output_file):
        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index,example) in enumerate(examples):
            feature = self.convert_single_example(ex_index,example,label_list,max_seq_length,tokenizer)
            #转成feature格式
            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict() 
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_id)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

    #保存的训练文件train.TFRecord，生成tf.data.TFRecordDataset input_fn供estimator.train()调用
    def file_based_input_fn_builder(self,input_file,seq_length,is_training,drop_remainder):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

            return d

        return input_fn
    #定义获取estimator实例的函数
    def get_estimator(self):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig

        bert_config = modeling.BertConfig.from_json_file(args.config_name)
        label_list = self.processor.get_labels()
        train_examples = self.processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / self.batch_size * args.num_train_epochs)
        #学习率预热系数
        num_warmup_steps = int(num_train_steps * 0.1)
        #模型框架初始值，如果是微调训练模型初始值使用bert中的initpoint如果是预测模型的参数使用output中的
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            init_checkpoint = args.ckpt_name
        else:
            init_checkpoint = args.output_dir
        #创建model_fn对象,该对象中定义了整个模型结构框架
        model_fn = self.model_fn_builder(bert_config=bert_config,num_labels=len(label_list),init_checkpoint=init_checkpoint,learning_rate=args.learning_rate,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps,use_one_hot_embeddings=False)

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
        strategy= tf.contrib.distribute.MirroredStrategy(devices=["/device:GPU:0","/device:GPU:1"])
        config_dis = tf.estimator.RunConfig(train_distribute=strategy)
        return Estimator(model_fn=model_fn,config=config_dis,model_dir=args.output_dir,params={'batch_size':self.batch_size})

    #定义训练模型函数
    def train(self):
        if self.mode is None:
            raise ValueError("没有发现模型参数信息")
        #从bert的模型文件中加载json文件
        bert_config = modeling.BertConfig.from_json_file(args.config_name)
        if args.max_seq_len > bert_config.max_position_embeddings:
            raise ValueError("设置的最大长度比bert模型的最大长度要长")
        #创建新模型输出目录
        tf.gfile.MakeDirs(args.output_dir)
        #获取类别
        label_list = self.processor.get_labels()
        #获取InputExamples格式的输入数据
        train_examples = self.processor.get_train_examples(args.data_dir)
        #设置总的训练次数:样本总数除以batchsize然后乘以训练轮数
        num_train_steps = int(len(train_examples)/args.batch_size*args.num_train_epochs)
        #比较关键的一步获取estimator对象
        estimator = self.get_estimator()
        #定义train.tf_record文件,用于存储train过程中inputexample和inputfeatures
        train_file = os.path.join(args.output_dir,"train.tf_record")
        self.file_based_convert_examples_to_features(train_examples,label_list,args.max_seq_len,self.tokenizer,train_file)
        tf.logging.info("****************running training ******************")
        tf.logging.info("输入样本数 = %d",len(train_examples))
        tf.logging.info("batch size = %d",args.batch_size)
        tf.logging.info("训练总次数 = %d",num_train_steps)
        #根据保存的训练文件train.TFRecord，生成tf.data.TFRecordDataset用于提供给Estimator来训练
        train_input_fn = self.file_based_input_fn_builder(input_file=train_file,seq_length=args.max_seq_len,is_training=True,drop_remainder=True)
        estimator.train(input_fn=train_input_fn,max_steps=num_train_steps)

    #针对迭代器从新定义example转feature函数
    def convert_examples_to_features(self,examples,label_list,max_seq_length,tokenizer):
        for (ex_index,example) in enumerate(examples):
            label_map={}
            for (i,label)in enumerate(label_list):
                label_map[label]=i

            textlist = example.text_a.split(' ')
            labellist = example.label.split(' ')
            tokens = []
            labels = []
            for i,word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m==0:
                        labels.append(label_1)
                    else:
                        labels.append("X")
            if len(tokens) > max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length -2)]

            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            label_ids.append(label_map["[CLS]"])
            for i,token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                label_ids.append(label_map[labels[i]])
            
            ntokens.append("[SEP]")
            segment_ids.append(0)
            label_ids.append(label_map["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                ntokens.append("**NULL**")

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_ids)

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
        return (tf.data.Dataset.from_generator(self.generate_from_queue,output_types={'input_ids': tf.int32,'input_mask': tf.int32,'segment_ids': tf.int32,'label_ids': tf.int32},output_shapes={'input_ids': (None, self.max_seq_length),'input_mask': (None, self.max_seq_length),'segment_ids': (None, self.max_seq_length),'label_ids': (None,self.max_seq_length)}).prefetch(10))

if __name__ == '__main__':
    cls = BertNer()
    cls.set_mode(tf.estimator.ModeKeys.TRAIN)
    cls.train()
    #开始预测
    '''   
    cls.set_mode(tf.estimator.ModeKeys.PREDICT)
    
    while True:
        inputsentence = input('输入句子:')
        length = len(inputsentence)
        cls.input_queue.put([inputsentence])
        predict = cls.output_queue.get()
        print (predict[0][1:length+1])
    '''
