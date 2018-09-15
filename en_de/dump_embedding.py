import numpy as np
from utils import PAD_ID, calAvgSimC, calMaxSimC
from random import sample, shuffle, random, randint
import tensorflow as tf
import sys, os
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
    argparser.add_argument("--sample_size",
            type = int,
            default = 25,
            help = "number of negative samples for skip-gram"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 2048,
            help = "batch size"
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

batch_size = args.batch_size
samp_size = args.sample_size
context_window = args.context_window
sense_dim = args.sense_number
embedding_dim = args.selection_dim
sense_embedding_dim = args.representation_dim
dataset_prefix = args.dataset_dir
ckpt_path = args.ckpt_path
lr_rate = args.lr_rate

# read in data
en_dh = data.data_helper(dataset_prefix, sense_dim, batch_size, context_window, 'en')
de_dh = data.data_helper(dataset_prefix, sense_dim, batch_size, context_window, 'de')
en_dh.set_bi_gen(de_dh.file_gen())
de_dh.set_bi_gen(en_dh.file_gen())

# construct embedding and model
en_emb = embedding(en_dh.vocab_size, en_dh.sense_size, embedding_dim, sense_embedding_dim, sense_dim)
de_emb = embedding(de_dh.vocab_size, de_dh.sense_size, embedding_dim, sense_embedding_dim, sense_dim)
en_ss = sense_selection(en_emb.w_in, en_emb.w_out, sense_dim, embedding_dim, batch_size, context_window, lr_rate, de_emb.w_in)
de_ss = sense_selection(de_emb.w_in, de_emb.w_out, sense_dim, embedding_dim, batch_size, context_window, lr_rate, en_emb.w_in)
en_srl = sense_representation_learning(en_emb.s_in, en_emb.s_out, en_dh.sense_counts, samp_size, lr_rate, de_dh.sense_counts, de_emb.s_out)
de_srl = sense_representation_learning(de_emb.s_in, de_emb.s_out, de_dh.sense_counts, samp_size, lr_rate, en_dh.sense_counts, en_emb.s_out)

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

    scores = [maxSimC, avgSimC]
    score_name = ['MaxSimC', 'AvgSimC']
    print 'scws AvgSimC =','{:.5f}'.format(avgSimC), 'MaxSimC =','{:.5f}'.format(maxSimC)
    ret_str = 'scws AvgSimC = {:.5f}'.format(avgSimC) + ' MaxSimC = {:.5f}'.format(maxSimC)
    return ret_str, scores, score_name

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # reload model
    best_saver.restore(sess, ckpt_path)
    evaluate_scws(sess, en_dh, batch_size, en_ss, en_srl)
    # dump embedding files
    en_emb, de_emb = sess.run([en_emb.s_in, de_emb.s_in])
    en_vocab = open(dataset_prefix+'en_vocab').read().splitlines()
    de_vocab = open(dataset_prefix+'de_vocab').read().splitlines()
    en_out = open('en_sense_emb', 'w')
    for idx, line in enumerate(en_vocab):
        word = line.split(' ')[0]
        for i in range(sense_dim):
            emb = en_emb[idx*sense_dim+i]
            emb = ' '.join(map(str, emb))
            en_out.write('{}_{} {}\n'.format(word, i, emb))
    en_out.close()
    de_out = open('de_sense_emb', 'w')
    for idx, line in enumerate(de_vocab):
        word = line.split(' ')[0]
        for i in range(sense_dim):
            emb = de_emb[idx*sense_dim+i]
            emb = ' '.join(map(str, emb))
            de_out.write('{}_{} {}\n'.format(word, i, emb))
    de_out.close()
