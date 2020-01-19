import tensorflow as tf
import numpy as np
import math
from tensorflow.python import debug as tf_debug
import tqdm
import copy

# def latent_combine(genes,kernel=5):

def feature_slice(input,gene_num,batch_size):
    feature_list=[]
    for gene_id in range(gene_num):
        tmp_input_1 = input[:,0:gene_id]
        tmp_input_2 = input[:,gene_id + 1:]
        gene=tf.concat([tmp_input_1,tmp_input_2],axis=1)
        gene=tf.reshape(gene,shape=[batch_size,-1])
        # gene=tf.nn.dropout(gene,keep_prob=0.7)
        feature_list.append(gene)
        # print(gene.shape)
    return feature_list

def positional_encoding(inputs,
                        maxlen,
                        masking=False,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

class static_network():
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
        self.train_split=True
        if config.model_type=='static':
            self.build()

        # self.beta_VAE_build()
    def pos_embedding(self,input):
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / self.emb_size) for i in range(self.emb_size)]
            for pos in range(self.max_time)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)
        outputs = tf.nn.embedding_lookup(position_enc, input)
        return outputs


    def beta_VAE_build(self):
        if self.use_label:
            self.input_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
            self.output_label=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.label_num])

            self.label_variable = tf.get_variable(name='label_value', shape=[self.label_num, self.emb_size],
                                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.label_emb = tf.nn.embedding_lookup(self.label_variable, self.input_label)

        with tf.variable_scope('static_net',reuse=tf.AUTO_REUSE):

            self.input=tf.placeholder(dtype=tf.float32,shape=[None,self.gene_num])
            self.output=tf.placeholder(dtype=tf.float32,shape=[None,self.gene_num])
            self.avg_input=tf.placeholder(dtype=tf.float32)
            self.std_input=tf.placeholder(dtype=tf.float32)


            self.batch_size_ph=tf.cast(self.batch_size,dtype=tf.float32)
            self.input_gene_value=tf.reshape(self.input,shape=[self.batch_size,self.gene_num])

            self.input_gene=tf.reshape(self.input,shape=[self.batch_size,self.gene_num,1])
            self.input_gene=tf.layers.dense(self.input_gene,units=self.hidden_size,use_bias=True)
            self.label_emb=tf.expand_dims(self.label_emb,axis=1)
            self.label_emb=tf.tile(self.label_emb,[1,self.gene_num,1])
            # print(self.input.shape,self.label_emb.shape)
            self.input_gene=tf.concat([self.input_gene,self.label_emb],axis=2)
            self.input_gene=tf.nn.dropout(self.input_gene,keep_prob=0.7)

            self.input_gene=tf.reshape(self.input_gene,shape=[self.batch_size,-1])
            # self.input_gene = tf.clip_by_value(self.input_gene, 1e-8, 1 - 1e-8)
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            self.input_hidden=tf.layers.dense(self.input_gene,units=self.gene_num*self.hidden_size,activation=tf.nn.relu)
            self.input_hidden=tf.nn.dropout(self.input_hidden,keep_prob=0.7)
            self.input_hidden=tf.layers.dense(self.input_hidden,units=self.gene_num*2,activation=tf.nn.relu)
            self.mean = self.input_hidden[:, :self.gene_num]
            self.stddev = 1e-6 + tf.nn.softplus(self.input_hidden[:, self.gene_num:])
            z = self.mean + self.stddev * tf.random_normal(tf.shape(self.mean), 0, 1, dtype=tf.float32)
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            self.output_hidden=tf.layers.dense(z,units=self.gene_num*self.hidden_size,activation=tf.nn.relu)
            self.output_gene=tf.layers.dense(self.output_hidden,units=self.gene_num,activation=tf.nn.elu)

            # self.output_gene=tf.clip_by_value(self.output_gene, 1e-8, 1 - 1e-8)
        marginal_likelihood = tf.reduce_sum(self.input_gene_value * tf.log(self.output_gene+1e-8) +
                                            (1 - self.input_gene_value) * tf.log(1 - self.output_gene+1e-8), 1)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mean) + tf.square(self.stddev)
                                            - tf.log(1e-8 + tf.square(self.stddev)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)
        # self.I=marginal_likelihood
        ELBO = marginal_likelihood - KL_divergence

        self.loss = -ELBO
        grads_and_vars = self.opt.compute_gradients(self.loss)
        self.train_op=self.opt.apply_gradients(grads_and_vars)
        tf.summary.scalar('loss', self.loss)
        self.sess = tf.Session()
        # self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sum_writer = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def build(self):
        print('static')
        self.opt_list=[]
        # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        for _ in range(self.gene_num):
            self.opt_list.append(tf.train.AdadeltaOptimizer(learning_rate=self.lr*0.1))
        self.opt_list[0] = tf.train.AdadeltaOptimizer(learning_rate=self.lr*0.1)    #init_loss==xxxx    #valid
        self.opt_list[1] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[2] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[3] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[4] = tf.train.AdadeltaOptimizer(1e-9 * 0.01)  # init_loss==XXXX
        self.opt_list[5] = tf.train.AdadeltaOptimizer(1e-9 * 0.01)  # init_loss==XXXX
        self.opt_list[6] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[7] = tf.train.AdadeltaOptimizer(1e-9 * 0.01)  # init_loss==XXXX
        self.opt_list[8] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[9] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[10] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[11] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[12] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[13] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[14] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[15] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[16] = tf.train.AdadeltaOptimizer(1e-9 * 0.01)  # init_loss==XXXX
        self.opt_list[17] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[18] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[19] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[20] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[21] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[22] = tf.train.AdadeltaOptimizer(1e-9 * 0.01)  # init_loss==XXXX
        self.opt_list[23] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[24] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[25] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[26] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[27] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[28] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[29] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[30] = tf.train.AdadeltaOptimizer(1e-9 * 0.01)  # init_loss==XXXX
        self.opt_list[31] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[32] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[33] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[34] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[35] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX
        self.opt_list[36] = tf.train.AdadeltaOptimizer(learning_rate=self.lr * 0.1)  # init_loss==XXXX

        with tf.variable_scope('hidden_feature'):
            self.input_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
            self.output_label=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.label_num])

            self.label_variable = tf.get_variable('label_var',shape=[self.label_num, self.emb_size],
                                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.label_emb = tf.nn.embedding_lookup(self.label_variable, self.input_label)

            self.time=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])
            self.time_pos=self.pos_embedding(self.time)
            self.time_pos = tf.expand_dims(self.time_pos, axis=1)
            self.time_pos = tf.tile(self.time_pos, [1, self.gene_num, 1])

            self.input=tf.placeholder(dtype=tf.float32,shape=[None,self.gene_num])
            self.output=tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.gene_num])


            self.batch_size_ph=tf.cast(self.batch_size,dtype=tf.float32)
            self.input_gene=tf.reshape(self.input,shape=[self.batch_size,self.gene_num,1])

            self.input_gene=tf.reshape(self.input_gene,shape=[self.batch_size,self.gene_num,1])
            self.input_gene=tf.layers.dense(self.input_gene,units=self.hidden_size,use_bias=True,activation=tf.nn.relu)
            # print(self.input_gene.name)
            self.label_emb=tf.expand_dims(self.label_emb,axis=1)
            self.label_emb=tf.tile(self.label_emb,[1,self.gene_num,1])
            if self.config.use_position_embedding:
            # print(self.input.shape,self.label_emb.shape)
                self.input_gene=tf.concat([self.input_gene,self.label_emb,self.time_pos],axis=2)
                self.input_gene=tf.layers.dense(self.input_gene,units=self.hidden_size,use_bias=True,activation=tf.nn.relu)
            else:
                self.input_gene = tf.concat([self.input_gene, self.label_emb], axis=2)
                self.input_gene=tf.layers.dense(self.input_gene,units=self.hidden_size,use_bias=True,activation=tf.nn.relu)
            self.input_gene=tf.nn.dropout(self.input_gene,keep_prob=0.7)

            self.hidden_input=feature_slice(self.input_gene,self.gene_num,self.batch_size)

        self.hidden_list1=[]

        self.hidden_list2=[]
        self.output_list=[]

        self.loss_list=[]
        self.total_loss=0

        self.train_op=[]
        self.gene_op=[]
        self.pre_input_output=tf.reshape(self.input_gene,shape=[self.batch_size,-1])
        self.output_tensor = tf.layers.dense(self.pre_input_output,self.gene_num)
        for i in range(self.gene_num):
            with tf.variable_scope('gene'+str(i)):
                gene=tf.layers.dense(self.hidden_input[i],units=self.hidden_size,
                                         use_bias=True,activation=tf.nn.relu)
                gene=tf.nn.dropout(gene,keep_prob=0.7)
                self.hidden_list2.append(tf.layers.dense
                                        (gene,units=self.hidden_size,
                                         use_bias=True,activation=tf.nn.relu))
                self.output_list.append(tf.layers.dense
                                        (self.hidden_list2[i],units=1,
                                         use_bias=True))
                # self.output_tensor=tf.concat([self.output_tensor,self.output_list[i]],axis=1)
                loss=(tf.losses.mean_squared_error(self.output[:,i:i+1],self.output_list[i]))/self.batch_size_ph
            gene_scope=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gene'+str(i))
            g_v_gene=self.opt_list[i].compute_gradients(loss,var_list=gene_scope)
            self.gene_op.append(self.opt_list[i].apply_gradients(g_v_gene))

            train_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gene'+str(i))
            # print(train_scope)
            self.loss_list.append(loss)
            self.total_loss+=loss
            grads_and_vars = self.opt.compute_gradients(loss)
            self.train_op.append(self.opt.apply_gradients(grads_and_vars))

        # self.all_loss=tf.losses.mean_squared_error(self.output,self.output_tensor)
        # # print(self.output_tensor.shape)
        # hidden_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hidden_feature')
        # # print(hidden_scope)
        # grads_and_vars=self.opt.compute_gradients(self.all_loss,var_list=hidden_scope)
        # self.pre_op=self.opt.apply_gradients(grads_and_vars)

        if self.config.pred_label:
            with tf.variable_scope('labels_output',reuse=tf.AUTO_REUSE):
                self.label_bias=tf.get_variable(name='label_outputs',shape=[self.batch_size,self.hidden_size])
                for i in range(self.gene_num):
                    self.label_bias=tf.concat([self.label_bias,self.hidden_list2[i]],axis=1)
                self.label_bias=tf.layers.dense(self.label_bias,units=self.hidden_size,
                                                use_bias=False,activation=tf.nn.relu)
                self.label_bias = tf.layers.dense(self.label_bias, units=self.hidden_size,
                                                  use_bias=False, activation=tf.nn.relu)
                self.logits = tf.layers.dense(self.label_bias, self.label_num)
                self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
                correct=tf.equal(tf.argmax(self.output_label, 1), self.y_pred_cls)
                self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.output_label)
                label_loss = tf.reduce_mean(cross_entropy)
                self.total_loss+=label_loss/self.batch_size_ph
            train_scope=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='labels_output')

            self.label_optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(label_loss,var_list=train_scope)


        tf.summary.scalar('loss',self.total_loss)
        self.sess=tf.Session()
        # self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
        init=tf.global_variables_initializer()
        self.sess.run(init)
        self.sum_writer=tf.summary.merge_all()
        self.writer=tf.summary.FileWriter(self.log_path,self.sess.graph)



    def train(self, train_data,valid_data):
        min_val_loss=10000
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, './data/complete/model/7')
        history_loss=10000
        for epoch in range(self.epochs):
            print(epoch)
            train_data.re_init()
            valid_data.re_init()
            train_loss=0
            val_loss=0
            train_label_acc=0
            val_label_acc=0
            for i,data in tqdm.tqdm(enumerate(train_data)):
                fetch_dict=self.fetch_gene(data,self.use_label)
                if self.train_split==False:
                    if self.config.pred_label:
                        writer,_,_,loss,acc=\
                            self.sess.run([self.sum_writer,self.train_op,self.label_optim,self.total_loss,self.acc],
                                          feed_dict=fetch_dict)

                        train_loss+=loss
                        train_label_acc+=acc
                    else:
                        writer, _, loss,loss_list = \
                            self.sess.run([self.sum_writer, self.train_op, self.total_loss,self.loss_list],
                                          feed_dict=fetch_dict)
                else:
                    writer, _, loss, loss_list = \
                        self.sess.run([self.sum_writer, self.gene_op, self.total_loss, self.loss_list],
                                      feed_dict=fetch_dict)
                    train_loss += loss
            print(loss_list)

                # self.writer.add_summary(writer)
            for i,data in tqdm.tqdm(enumerate(valid_data)):
                fetch_dict=self.fetch_gene(data,self.use_label)
                if self.config.pred_label:
                    writer,loss,acc = self.sess.run([self.sum_writer,self.total_loss,self.acc], feed_dict=fetch_dict)
                else:
                    writer, loss,loss_list = self.sess.run([self.sum_writer, self.total_loss,self.loss_list],
                                                      feed_dict=fetch_dict)
                val_loss += loss
                self.writer.add_summary(writer)
            print('valid')
            print(loss_list)


            train_loss/=len(train_data)
            val_loss/=len(valid_data)
            if epoch==6:
                self.train_split=True
                print('now split training')
            else:
                history_loss=train_loss
            if self.config.pred_label:
                train_label_acc/=(len(train_data)/self.batch_size)
                val_label_acc /= (len(valid_data) / self.batch_size)
                print(train_label_acc,val_label_acc)
            if val_loss<min_val_loss:
                min_val_loss=val_loss
            self.saver.save(self.sess,self.model_path+str(epoch))
            print(train_loss,val_loss)

    def test(self,test_data,choose_id=None,model_loader=True):
        if model_loader:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)
        output_list=[[] for _ in range(len(choose_id))]

        for i, data in tqdm.tqdm(enumerate(test_data)):
            fetch_dict = self.fetch_gene(data,use_label=self.use_label,mode='test')
            if choose_id==None:
                output_list = self.sess.run([self.output_list], feed_dict=fetch_dict)
            else:
                for idx,id in enumerate(choose_id):
                    batch_output = self.sess.run([self.output_list[id]],feed_dict=fetch_dict)
                    output_list[idx].append(batch_output)
        for i in range(len(output_list)):
            output_list[i]=np.array(output_list[i])
            output_list[i]=output_list[i].reshape((-1,1))
            output_list[i]=output_list[i][:test_data.data_num]
        return output_list

    def fetch_gene(self,data, use_label,mode='train'):
        if self.config.use_position_embedding:
            label=data[:,0].astype('int32')
            time=data[:,1].astype('int32')
            data=data[:,2:]
            if mode=='train':
                return {self.input_label: label,self.time:time, self.input: data, self.output: data}
            else:
                return {self.input_label: label,self.time:time, self.input: data}
        if use_label:
            label = data[:, 0].astype('int32')
            pred_label=np.zeros([self.batch_size,self.label_num])
            for i in range(self.batch_size):
                pred_label[i][label[i]]=1
            data = data[:, 1:]
            fetch_dict = {self.input_label: label, self.input: data, self.output: data,self.output_label:pred_label}
            if mode=='train':
                return fetch_dict
            else:
                return {self.input_label: label, self.input: data}
        else:
            if mode=='train':
                return {self.input: data, self.output: data}
            else:
                return {self.input: data}

class dynamic_network(static_network):
    def __init__(self,config):
        super(dynamic_network, self).__init__(config)
        self.max_length=config.max_length
        self.build()

    def build(self):

        self.input_ph=tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.max_length,self.gene_num+2])
        self.input_value=tf.concat([self.input_ph[:,0:1,:],self.input_ph],axis=1)
        self.label=tf.cast(self.input_value[:,:,0:1],dtype=tf.int32)
        self.time=tf.cast(self.input_value[:,:,1:2],dtype=tf.int32)
        self.input=self.input_value[:,:,2:]
        # print(self.label.shape)
        initial_variable=tf.get_variable(name='start',shape=[self.batch_size,1,self.gene_num],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        end_variable = tf.get_variable(name='end', shape=[self.batch_size, 1, self.gene_num],
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.output_gene=tf.reshape(self.input_ph[:,:,2:],shape=[self.batch_size,self.max_length,self.gene_num])
        self.output_gene=tf.concat([self.output_gene,self.output_gene[:,self.max_length-1:self.max_length,:]],axis=1)
        self.output_gene=tf.subtract(self.output_gene,self.input)
        self.output_gene=self.output_gene[:,1:self.max_length,:]
        self.output_gene=tf.concat([initial_variable,self.output_gene,end_variable],axis=1)
        # self.output_gene[:,0:1,:]=initial_variable
        # self.output_gene[:,self.max_length:self.max_length+1,:]=end_variable

        self.label_variable = tf.get_variable(name='label_value', shape=[self.label_num, self.emb_size],
                                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        self.label_emb = tf.nn.embedding_lookup(self.label_variable, self.label)
        # self.label_emb=tf.reshape(self.label_emb,shape=[self.batch_size,self.max_length,self.label_emb])
        self.label_emb=tf.squeeze(self.label_emb,axis=2)
        # print(self.label_emb.shape)

        self.input_gene = tf.reshape(self.input, shape=[self.batch_size, self.max_length+1,self.gene_num, 1])
        self.input_gene = tf.layers.dense(self.input_gene, units=self.hidden_size, use_bias=True)
        # self.label_emb = tf.expand_dims(self.label_emb, axis=2)
        # self.label_emb = tf.tile(self.label_emb, [1, 1,self.gene_num, 1])
        # print(self.label_emb.shape)

        self.time_pos = self.pos_embedding(self.time)
        self.time_pos=tf.squeeze(self.time_pos,axis=2)
        # print(self.time_pos.shape)
        # self.time_pos = tf.expand_dims(self.time_pos, axis=2)
        # self.time_pos = tf.tile(self.time_pos, [1,1,self.gene_num, 1])
        # print(self.time_pos.shape)

        # print(self.input.shape,self.label_emb.shape)
        self.input_gene=tf.reshape(self.input_gene,shape=[self.batch_size,self.max_length+1,-1])
        self.input_gene = tf.concat([self.input_gene, self.label_emb, self.time_pos], axis=2)
        self.input_gene = tf.nn.dropout(self.input_gene, keep_prob=0.7)
        # print(self.input_gene.shape)
        lstm_cells = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size)
        self.outputs, states = tf.nn.dynamic_rnn(lstm_cells, self.input_gene, dtype=tf.float32)
        self.pred_out=tf.layers.dense(self.outputs,units=self.gene_num,activation=tf.nn.elu)
        print(self.pred_out.shape)
        self.loss = tf.losses.mean_squared_error(self.output_gene,self.pred_out)
        grads_and_vars = self.opt.compute_gradients(self.loss)
        self.train_op=self.opt.apply_gradients(grads_and_vars)

        # print(self.outputs.shape)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, train_data,valid_data):
        min_val_loss=10000
        for epoch in range(self.epochs):
            train_data.re_init()
            valid_data.re_init()
            train_loss=np.zeros(self.gene_num)
            val_loss=np.zeros(self.gene_num)
            train_label_acc=0
            val_label_acc=0
            for i,data in tqdm.tqdm(enumerate(train_data)):
                fetch_dict={self.input_ph:data}
                loss,_ = \
                    self.sess.run([self.loss,self.train_op],
                                  feed_dict=fetch_dict)
                train_loss+=loss
                # print(check)
            #
            for i,data in tqdm.tqdm(enumerate(valid_data)):
                fetch_dict={self.input_ph:data}
                loss = self.sess.run([self.loss],feed_dict=fetch_dict)
                val_loss+=loss
            #     self.writer.add_summary(writer)
            #
            # train_loss/=len(train_data)
            # val_loss/=len(valid_data)
            # if self.config.pred_label:
            #     train_label_acc/=(len(train_data)/self.batch_size)
            #     val_label_acc /= (len(valid_data) / self.batch_size)
            #     print(train_label_acc,val_label_acc)

            self.saver.save(self.sess,self.model_path+str(epoch))
            print(np.mean(train_loss)/len(train_data),np.mean(val_loss)/len(valid_data))
    # def fetch_gene(self,data, use_label,mode='train'):
    #     label = data[:,:, 0].astype('int32')
    #     time = data[:,:, 1].astype('int32')
    #     data = data[:,:, 2:]



