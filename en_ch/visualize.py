import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

en_sense = open('en_sense_emb').read().splitlines()
ch_sense = open('ch_sense_emb').read().splitlines()
emb_mat = []
emb_word = []
for line in en_sense:
    word = line.split(' ')[0]+'_e'
    emb = list(map(float, line.split(' ')[1:]))
    emb_mat.append(emb)
    emb_word.append(word)

for line in ch_sense:
    word = line.split(' ')[0]+'_c'
    emb = list(map(float, line.split(' ')[1:]))
    emb_mat.append(emb)
    emb_word.append(word)

sess = tf.Session()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=[len(emb_mat), 300])
set_x = tf.assign(X, place, validate_shape=False)

sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: emb_mat})

# write labels
with open('tb/metadata.tsv', 'w') as f:
    for word in emb_word:
        line_out = "%s\n" % word
        f.write(line_out)#.encode('utf-8'))

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter('tb/emb_viz_%s.log' % 1, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = '../metadata.tsv'
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
saver.save(sess, "tb/model.ckpt")
