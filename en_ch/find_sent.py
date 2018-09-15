import numpy as np
from utils import PAD_ID, calAvgSimC, calMaxSimC
from random import sample, shuffle, random, randint
from numpy.random import choice
import tensorflow as tf
import sys, os
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
import data
from model import sense_selection, sense_representation_learning
from emb import embedding

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--lr_rate",
            type = float,
            default = 0.025,
            help = "learning rate"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 512,
            help = "batch size"
        )
    argparser.add_argument("--sample_size",
            type = int,
            default = 25,
            help = "number of negative samples for skip-gram"
        )
    argparser.add_argument("--context_window",
            type = int,
            default = 5,
            help = "context window size"
        )
    argparser.add_argument("--sense_number",
            type = int,
            default = 3,
            help = "sense number per word"
        )
    argparser.add_argument("--selection_dim",
            type = int,
            default = 300,
            help = "embedding dimension for the selection module"
        )
    argparser.add_argument("--representation_dim",
            type = int,
            default = 300,
            help = "embedding dimension for the representation module"
        )
    argparser.add_argument("--dataset_dir",
            type = str,
            required = True,
            help = "directory containing the dataset"
        )
    argparser.add_argument("--memory",
            type = float,
            default = 0.1,
            help = "GPU memory fraction used for model training"
        )
    argparser.add_argument("--ckpt_path",
            type = str,
            required = True,
            help = "path to checkpoint files"
        )
    args = argparser.parse_args()
    return args

args = load_arguments()

lr_rate = args.lr_rate
batch_size = args.batch_size
samp_size = args.sample_size
context_window = args.context_window
sense_dim = args.sense_number
embedding_dim = args.selection_dim
sense_embedding_dim = args.representation_dim
dataset_prefix = args.dataset_dir
ckpt_path = args.ckpt_path

# read in data
en_dh = data.data_helper(dataset_prefix, sense_dim, batch_size, context_window, 'en')
ch_dh = data.data_helper(dataset_prefix, sense_dim, batch_size, context_window, 'ch')
en_dh.set_bi_gen(ch_dh.file_gen())
ch_dh.set_bi_gen(en_dh.file_gen())

# construct embedding and model
en_emb = embedding(en_dh.vocab_size, en_dh.sense_size, embedding_dim, sense_embedding_dim, sense_dim)
ch_emb = embedding(ch_dh.vocab_size, ch_dh.sense_size, embedding_dim, sense_embedding_dim, sense_dim)
en_ss = sense_selection(en_emb.w_in, en_emb.w_out, sense_dim, embedding_dim, batch_size, context_window, lr_rate, ch_emb.w_in)
ch_ss = sense_selection(ch_emb.w_in, ch_emb.w_out, sense_dim, embedding_dim, batch_size, context_window, lr_rate, en_emb.w_in)
en_srl = sense_representation_learning(en_emb.s_in, en_emb.s_out, en_dh.sense_counts, samp_size, lr_rate, ch_dh.sense_counts, ch_emb.s_in)
ch_srl = sense_representation_learning(ch_emb.s_in, ch_emb.s_out, ch_dh.sense_counts, samp_size, lr_rate, en_dh.sense_counts, en_emb.s_in)
init = tf.global_variables_initializer()
best_saver = tf.train.Saver(tf.all_variables())

def getEmbedding(sess, dh, batch_size, ss, sense_mat, selected_indices, scws=False):
    if scws == True:
        testData = dh.scws_testData
        testLocation = dh.scws_testLocation
    else:
        testData = dh.testData
        testLocation = dh.testLocation
    senseVec = [[] for i in xrange(len(testData))]
    senseScore = [[] for i in xrange(len(testData))]
    currentIndex = 0
    while currentIndex < len(testData):
        c_indices = []
        while len(c_indices) != batch_size and currentIndex != len(testData):
            c_indices.append(testData[currentIndex])
            currentIndex+=1

        if len(c_indices) != batch_size and currentIndex == len(testData):
            while len(c_indices) != batch_size:
                c_indices.append(c_indices[-1])
                currentIndex += 1
        # context_window*2+batch_size
        c_indices += [c_indices[-1] for i in xrange(context_window*2)]
        sense_probability = sess.run(ss.sense_prob, feed_dict={
                                                        ss.context_indices: c_indices,
                                                        ss.eval_mode:True,
                                                        ss.bi_info:tf.SparseTensorValue([[0,0]], [0], [1,1]),
                                                        ss.lengths:[0 for _ in range(context_window*2+batch_size)],
                                                        ss.major_weight:0.0})
        for k in xrange(sense_dim):
            selected_s_input_indices = []
            for i in xrange(batch_size):
                if currentIndex-batch_size+i < len(testData):
                    wordIndex = testData[currentIndex-batch_size+i][testLocation[currentIndex-batch_size+i]]
                    selected_s_input_indices.append(wordIndex*sense_dim+ k)
                else:
                    selected_s_input_indices.append(0)
            embedded_sense = sess.run(sense_mat, feed_dict={selected_indices: selected_s_input_indices})

            for i in xrange(batch_size):
                index = currentIndex - batch_size + i
                if index < len(testData):
                    senseVec[index].append(embedded_sense[i])

        selected_w_input_indices = []
        for i in xrange(batch_size):
            if currentIndex - batch_size + i < len(testData):
                wordIndex = testData[currentIndex-batch_size+i][testLocation[currentIndex-batch_size+i]]
                selected_w_input_indices.append(wordIndex)
            else:
                selected_w_input_indices.append(0)
        for i in xrange(batch_size):
            index = currentIndex-batch_size+i
            if index < len(testData):
                senseScore[index] = sense_probability[i]

    return np.asarray(senseVec), np.asarray(senseScore)

def evaluate_scws(sess, en_dh, batch_size, en_ss, en_srl):
    global global_maxSimC
    global global_avgSimC

    [senseVec, senseScore] = getEmbedding(sess, en_dh, batch_size, en_ss, en_srl.embedded_sense_input, en_srl.selected_sense_input_indices, scws=True)

    indices1 = range(0, len(senseVec), 2)
    indices2 = range(1, len(senseVec), 2)

    avgSimC = calAvgSimC(en_dh.scws_test_score, senseVec[indices1], senseScore[indices1], senseVec[indices2], senseScore[indices2])
    maxSimC = calMaxSimC(en_dh.scws_test_score, senseVec[indices1], senseScore[indices1], senseVec[indices2], senseScore[indices2])

    print ('scws AvgSimC =','{:.5f}'.format(avgSimC), 'MaxSimC =','{:.5f}'.format(maxSimC))

def evaluate_bcws(sess, en_dh, ch_dh, batch_size, en_ss, ch_ss, en_srl, ch_srl):
    global global_bmaxSimC
    global global_bavgSimC
    
    [en_senseVec, en_senseScore] = getEmbedding(sess, en_dh, batch_size, en_ss, en_srl.embedded_sense_input, en_srl.selected_sense_input_indices)
    [ch_senseVec, ch_senseScore] = getEmbedding(sess, ch_dh, batch_size, ch_ss, ch_srl.embedded_sense_input, ch_srl.selected_sense_input_indices)
    
    assert en_dh.test_score == ch_dh.test_score
    avgSimC = calAvgSimC(en_dh.test_score, en_senseVec, en_senseScore, ch_senseVec, ch_senseScore)
    maxSimC = calMaxSimC(en_dh.test_score, en_senseVec, en_senseScore, ch_senseVec, ch_senseScore)

    print ('bcws AvgSimC =','{:.5f}'.format(avgSimC), 'MaxSimC =','{:.5f}'.format(maxSimC))

def init_topK(K, emb_in, emb_out):
    source_index = tf.placeholder(tf.int32)
    source_vec = tf.nn.l2_normalize(tf.nn.embedding_lookup(emb_in, source_index), axis=0)
    #source_vec = tf.nn.embedding_lookup(emb_in, source_index)
    source_vec = tf.expand_dims(source_vec, axis=0)
    dot_prod = tf.reduce_sum(tf.multiply(source_vec, tf.nn.l2_normalize(emb_out, axis=1)), axis=1)
    #dot_prod = tf.reduce_sum(tf.multiply(source_vec, emb_out), axis=1)
    values, indices = tf.nn.top_k(dot_prod, k=K)
    return source_index, indices

def topK(source_word, lang, K):
    if lang == 'en':
        source_emb = en_emb
        bi_emb = ch_emb
        dh = en_dh
        bi_dh = ch_dh

    elif lang == 'ch':
        source_emb = ch_emb
        bi_emb = en_emb
        dh = ch_dh
        bi_dh = en_dh

    si, topk = init_topK(K, source_emb.s_in, source_emb.s_in)
    bi_si, bi_topk = init_topK(K, source_emb.s_in, bi_emb.s_in)
    for s in range(sense_dim):
        res = sess.run(topk, feed_dict={si:dh.word2id[source_word]*sense_dim+s})
        ret = []
        for r in res:
            ret.append(dh.id2word[r/3]+'_'+str(r%3))
        print (ret)
    
    for s in range(sense_dim):
        res = sess.run(bi_topk, feed_dict={bi_si:dh.word2id[source_word]*sense_dim+s})
        ret = []
        for r in res:
            bi_word = bi_dh.id2word[r/3]+'_'+str(r%3)
            ret.append(bi_word)
        print str(ret).decode('string_escape')
        #print (ret)

def decode(sent, idx, lang):
    if lang == 'en':
        ss = en_ss
        dh = en_dh
    else:
        ss = ch_ss
        dh = ch_dh
    
    test_ids = []
    for word in sent:
        if word in dh.word2id:
            test_ids.append(dh.word2id[word])
        else:
            test_ids.append(dh.word2id['_UNK'])
    left = []
    right = []
    if idx < context_window:
        left = [PAD_ID]*(context_window-idx)
    if idx + context_window + 1 > len(test_ids):
        right = (idx+context_window+1-len(test_ids))*[PAD_ID]
    tar_indices = left + test_ids[max(0,idx-context_window):idx+context_window+1] + right
    c_indices = []
    for _ in range(context_window*2+batch_size):
        c_indices.append(tar_indices)
    sense_probability = sess.run(ss.sense_prob, feed_dict={
                                                ss.context_indices: c_indices,
                                                ss.eval_mode:True,
                                                ss.bi_info:tf.SparseTensorValue([[0,0]], [0], [1,1]),
                                                ss.lengths:[0 for _ in range(context_window*2+batch_size)],
                                                ss.major_weight:0.0})
    print (sense_probability[0].tolist())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # reload model
    best_saver.restore(sess, ckpt_path)
    #evaluate_bcws(sess, en_dh, ch_dh, batch_size, en_ss, ch_ss, en_srl, ch_srl)
    evaluate_scws(sess, en_dh, batch_size, en_ss, en_srl)
    while True:
        sent = raw_input('Please enter a sentence and the index you want to decode sense for:')
        sent = sent.split('\t')
        idx = int(sent[1])
        sent = sent[0].split(' ')
        word = sent[idx]
        if word in en_dh.word2id: # this target word is english
            decode(sent, idx, 'en')
            topK(word, 'en', 10)
        else:
            print ('This word is not in the vocabulary')
            
