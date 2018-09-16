import tensorflow as tf
import os
word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

class sense_selection(object):
    def dense_lookup(self, w_in, context_indices):
        # [(context_window*2+batch_size), max_context_length, embedding_dim]
        embedded_context = tf.nn.embedding_lookup(w_in, context_indices)
        embedded_context = tf.transpose(embedded_context, perm = [0,2,1])
        embedded_context = tf.reduce_mean(embedded_context, 2, keep_dims=True)
        return embedded_context

    def sparse_lookup(self, w_in, inputs_sparse, lengths):
        sparse_emb = tf.nn.embedding_lookup(w_in, inputs_sparse)
        mask = tf.cast(tf.sequence_mask(lengths, dtype=tf.bool), tf.float32)
        mask = tf.expand_dims(mask, axis=2)
        sparse_emb = tf.multiply(sparse_emb, mask)
        sparse_emb = tf.transpose(sparse_emb, perm=[0,2,1])
        sparse_emb = tf.reduce_mean(sparse_emb, 2, keep_dims=True)
        return sparse_emb

    def __init__(self, w_in, w_out, sense_dim, embedding_dim, batch_size, context_window, learning_rate, bi_w_in):
        max_context_length = 2*context_window + 1
        eval_mode = tf.placeholder(tf.bool, shape=[])
        self.eval_mode = eval_mode
        self.bi_info = tf.sparse_placeholder(tf.int32)
        bi_info = tf.sparse_to_dense(self.bi_info.indices, self.bi_info.dense_shape, self.bi_info.values)
        self.lengths = tf.placeholder(tf.int32, [context_window*2+batch_size])
        # add self here so we can feed it outside this class
        context_indices = tf.placeholder(tf.int32, [context_window*2+batch_size, max_context_length])
        sense_indices = tf.placeholder(tf.int32, [(context_window*2+batch_size) * sense_dim])
        self.context_indices = context_indices
        self.sense_indices = sense_indices
        major_weight = tf.placeholder(tf.float32)
        reg_weight = tf.placeholder(tf.float32)
        self.major_weight = major_weight
        self.reg_weight = reg_weight
        embedded_context = self.dense_lookup(w_in, context_indices)
        bi_embedded_context = self.sparse_lookup(bi_w_in, bi_info, self.lengths)

        # Combine bilingual contextual information
        embedded_context = tf.cond(eval_mode, lambda:tf.identity(embedded_context), lambda:tf.add(major_weight*embedded_context, (1-major_weight)*bi_embedded_context))

        # [(context_window*2+batch_size), sense_dim, embedding_dim]
        embedded_word_output = tf.nn.embedding_lookup(w_out, context_indices[:, context_window])
        
        # shape = [(context_window*2+batch_size), sense_dim, 1]
        sense_score = tf.matmul(embedded_word_output, embedded_context)
        
        # [(context_window*2+batch_size), sense_dim]
        sense_score = tf.squeeze(sense_score)

        # [context_window*2+batch_size]
        sense_greedy = tf.argmax(sense_score, 1)
        self.sense_greedy = sense_greedy
        
        target_sense_sampled_indices = tf.placeholder(tf.int32, [batch_size])
        self.target_sense_sampled_indices = target_sense_sampled_indices 
        # [batch_size]
        reward_prob = tf.placeholder(tf.float32, [batch_size], name='reward_logit')
        self.reward_prob = reward_prob
        
        # [(context_window*2+batch_size), sense_dim]
        sense_prob = tf.nn.softmax(sense_score)
        self.sense_prob = sense_prob
        entropy = -tf.multiply(tf.log(sense_prob+1e-8), sense_prob)
        entropy = tf.reduce_sum(entropy)*reg_weight
        # [(context_window*2+batch_size)* sense_dim]
        sense_score = tf.reshape(sense_score, [(context_window*2+batch_size)*sense_dim])
        # [batch_size]
        sense_selected_logit_input = tf.gather(sense_score, target_sense_sampled_indices)

        # [batch_size, sense_dim]
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=sense_selected_logit_input, labels=reward_prob))
        cost += entropy
        self.print_cost = cost
        self.print_ent = entropy
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.update = optimizer.minimize(cost)
    
class sense_representation_learning(object):
    def __init__(self, s_in, s_out, sense_counts, samp_size, learning_rate, bi_sense_counts, bi_s_mat):
        with tf.device("/cpu:0"):
            selected_sense_output_indices = tf.placeholder(tf.int32, None, name='selected_sense_output_indices')
            selected_sense_input_indices = tf.placeholder(tf.int32, None, name='selected_sense_input_indices')
            selected_bi_sense_output_indices = tf.placeholder(tf.int32, None, name='selected_bi_sense_output_indices')
            self.selected_sense_input_indices = selected_sense_input_indices
            self.selected_sense_output_indices = selected_sense_output_indices
            self.selected_bi_sense_output_indices = selected_bi_sense_output_indices

            # [batch_size, sense_dim, sense_embedding_dim]
            embedded_sense_input = tf.nn.embedding_lookup(s_in, selected_sense_input_indices)
            self.embedded_sense_input = embedded_sense_input
            embedded_sense_output = tf.nn.embedding_lookup(s_out, selected_sense_output_indices)
            self.embedded_sense_output = embedded_sense_output
            bi_embedded_sense_output = tf.nn.embedding_lookup(bi_s_mat, selected_bi_sense_output_indices)

            self.reward_sense_prob = tf.sigmoid(tf.reduce_sum(tf.multiply(embedded_sense_input, embedded_sense_output), 1))
            self.bi_reward_sense_prob = tf.sigmoid(tf.reduce_sum(tf.multiply(embedded_sense_input, bi_embedded_sense_output), 1))

            self.train = word2vec.neg_train_word2vec(s_in,s_out,selected_sense_input_indices,selected_sense_output_indices,\
              learning_rate, vocab_count=sense_counts, num_negative_samples=samp_size)

            # now to train the bilingual skip-gram
            self.train_bilingual = word2vec.neg_train_word2vec(s_in, bi_s_mat,\
                    selected_sense_input_indices,selected_bi_sense_output_indices, \
                    learning_rate, vocab_count=bi_sense_counts, num_negative_samples=samp_size)
