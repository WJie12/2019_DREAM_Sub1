import tensorflow as tf
class pretrain_model():
    def __init__(self,config):
        self.config=config
        self.batch_size=config.batch_size
        self.gene_num=config.gene_num
        self.hidden_size=config.hidden_size
        self.emb_size=config.label_emb_size
        self.lr=config.lr
        self.epochs=config.epoch
        self.model_path=config.model_dir
        self.log_path=config.log_dir
        self.use_label=config.use_label
        self.label_num=config.label_num
        self.max_time=config.max_time
        self.max_length=config.max_length
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

    def build(self):
        print('pre_trian_model')
        with tf.variable_scope('hidden'):
            self.input_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
            self.output_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.label_num])

            self.label_variable = tf.get_variable('label_var', shape=[self.label_num, self.emb_size],
                                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.label_emb = tf.nn.embedding_lookup(self.label_variable, self.input_label)
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.gene_num])
            self.output = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.gene_num])
            self.input_gene = tf.reshape(self.input, shape=[self.batch_size, self.gene_num, 1])

            self.input_gene = tf.reshape(self.input_gene, shape=[self.batch_size, self.gene_num, 1])
            self.input_gene = tf.layers.dense(self.input_gene, units=self.hidden_size, use_bias=True,
                                              activation=tf.nn.relu)

