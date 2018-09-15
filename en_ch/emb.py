import tensorflow as tf

class embedding(object):
    def __init__(self, vocab_size, sense_size, embedding_dim, sense_embedding_dim, sense_dim, bi_emb=None):
        if bi_emb == None:
            self.w_in = tf.Variable(tf.random_uniform([vocab_size, embedding_dim],-(3./embedding_dim)**0.5,(3./embedding_dim)**0.5),\
            trainable=True, name="word_embeddings")
            #self.w_out = tf.Variable(tf.zeros([vocab_size, sense_dim, embedding_dim]), trainable=True, name="word_outputs")
            self.w_out = tf.Variable(tf.random_uniform([vocab_size, sense_dim, embedding_dim],-(3./embedding_dim)**0.5,(3./embedding_dim)**0.5),\
            trainable=True, name="word_outputs")

            with tf.device("/cpu:0"):
                self.s_in = tf.Variable(tf.random_uniform([sense_size, sense_embedding_dim],-(3./sense_embedding_dim)**0.5,(3./sense_embedding_dim)**0.5),\
                              trainable=True, name="sense_embeddings")
                #self.s_out = tf.Variable(tf.zeros([sense_size, sense_embedding_dim]), trainable=True, name="sense_outputs")
                self.s_out = tf.Variable(tf.random_uniform([sense_size, sense_embedding_dim],-(3./sense_embedding_dim)**0.5,(3./sense_embedding_dim)**0.5),\
                              trainable=True, name="sense_outputs")
        
        elif bi_emb != None:
            print ('copy value!!!')
            self.w_in = tf.Variable(bi_emb.w_in.initialized_value())
            self.w_out = tf.Variable(bi_emb.w_out.initialized_value())
            with tf.device("/cpu:0"):
                self.s_in = tf.Variable(bi_emb.s_in.initialized_value())
                self.s_out = tf.Variable(bi_emb.s_out.initialized_value())
