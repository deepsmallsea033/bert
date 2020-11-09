import modeling
import tokenization
import args
from queue import Queue
from threading import Thread
import tensorflow as tf
import os

class InputExample(object):

    def __init__(self, unique_id, text_a):
        self.unique_id = unique_id
        self.text_a = text_a


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


class BertVector:

    def __init__(self, batch_size=1):
        """
        init BertVector
        :param batch_size:     Depending on your memory default is 32
        """
        self.max_seq_length = args.max_seq_len
        self.layer_indexes = args.layer_indexes
        self.gpu_memory_fraction = 1
        if os.path.exists(args.graph_file):
            self.graph_path = args.graph_file
        else:
            self.graph_path = optimize_graph()

        self.tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = self.get_estimator()
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.predict_thread.start()

    def create_model(self,bert_config,input_ids,input_mask,input_type_ids,use_one_hot_embeddings):
        model = modeling.BertModel(config=bert_config,is_training=False,input_ids=input_ids,input_mask=input_mask,token_type_ids=input_type_ids,use_one_hot_embeddings=use_one_hot_embeddings) 
        with tf.variable_scope("pooling"):
            if len(args.layer_indexes) == 1:
                encoder_layer = model.all_encoder_layers[args.layer_indexes[0]]
            else:
                all_layers = [model.all_encoder_layers[l] for l in args.layer_indexes]
                encoder_layer = tf.concat(all_layers, -1)
        mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
        masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10) 
        input_mask = tf.cast(input_mask, tf.float32)
        pooled = masked_reduce_mean(encoder_layer, input_mask)  
        pooled = tf.identity(pooled, 'final_encodes')
        output_tensors = [pooled]
        return output_tensors
    
    def model_fn_builder(self,bert_config,init_checkpoint,use_one_hot_embeddings):
        def model_fn(features,mode,params):
            from tensorflow.python.estimator.model_fn import EstimatorSpec
            input_ids = features["input_ids"]
            input_mask =features["input_mask"]
            input_type_ids =features["input_type_ids"]
            out_vec = self.create_model(bert_config,input_ids,input_mask,input_type_ids,use_one_hot_embeddings)
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            if init_checkpoint:
                (assignment_map, initialized_variable_names) \
                    = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            output_spec = EstimatorSpec(mode=mode, predictions={'encodes': out_vec[0]})
            return output_spec
        return model_fn

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
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config),model_dir=args.model_dir,
                         params={'batch_size': self.batch_size})

    def predict_from_queue(self):
        prediction = self.estimator.predict(input_fn=self.queue_predict_input_fn, yield_single_examples=False)
        for i in prediction:
            self.output_queue.put(i)

    def encode(self, sentence):
        self.input_queue.put(sentence)
        prediction = self.output_queue.get()['encodes']
        return prediction

    def queue_predict_input_fn(self):

        return (tf.data.Dataset.from_generator(
            self.generate_from_queue,
            output_types={'unique_ids': tf.int32,
                          'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32},
            output_shapes={
                'unique_ids': (None,),
                'input_ids': (None, self.max_seq_length),
                'input_mask': (None, self.max_seq_length),
                'input_type_ids': (None, self.max_seq_length)}).prefetch(10))

    def generate_from_queue(self):
        while True:
            features = list(self.convert_examples_to_features(seq_length=self.max_seq_length, tokenizer=self.tokenizer))
            yield {
                'unique_ids': [f.unique_id for f in features],
                'input_ids': [f.input_ids for f in features],
                'input_mask': [f.input_mask for f in features],
                'input_type_ids': [f.input_type_ids for f in features]
            }

    def convert_examples_to_features(self, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        input_masks = []
        examples = self._to_example(self.input_queue.get())
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            # if the sentences's length is more than seq_length, only use sentence's left part
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            # Where "input_ids" are tokens's index in vocabulary
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            input_masks.append(input_mask)
            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (example.unique_id))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            yield InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def _to_example(sentences):
        import re
        """
        sentences to InputExample
        :param sentences: list of strings
        :return: list of InputExample
        """
        unique_id = 0
        for ss in sentences:
            line = tokenization.convert_to_unicode(ss)
            if not line:
                continue
            line = line.strip()
            text_a = line
            yield InputExample(unique_id=unique_id, text_a=text_a)
            unique_id += 1


if __name__ == "__main__":
    bert = BertVector()
    while True:
        input_str = input()
        v = bert.encode([input_str])
        print(str(v[0]))
