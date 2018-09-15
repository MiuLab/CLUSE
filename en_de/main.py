import numpy as np
from utils import calAvgSimC, calMaxSimC
from random import sample, shuffle, random, randint
from numpy.random import choice
import tensorflow as tf
import sys, os
import argparse
import data
from model import sense_selection, sense_representation_learning
from emb import embedding

tf.set_random_seed(20)
np.random.seed(20)

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--major_weight",
            type = float,
            required=True,
            help = "weight for major contextual vector"
        )
    argparser.add_argument("--reg_weight",
            type = float,
            required=True,
            help = "weight for entropy regularization"
        )
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
    argparser.add_argument("--save_dir",
            type = str,
            required = True,
            help = "directory for saving checkpoints (warning: lots of space will be consumed)"
        )
    argparser.add_argument("--log_path",
            type = str,
            required = True,
            help = "path for saving training logs"
        )
    argparser.add_argument("--memory",
            type = float,
            default = 0.1,
            help = "GPU memory fraction used for model training"
        )
    argparser.add_argument("--ckpt_threshold",
            type = float,
            default = 0.6,
            help = "start checkpoint saving for best models (MaxSimC and AvgSimC) after the threshold is reached."
        )
    argparser.add_argument("--training_epochs",
            type = int,
            default = 100,
            help = "max training epochs"
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
save_dir = args.save_dir
log_path = args.log_path
training_epochs = args.training_epochs
ckpt_threshold = args.ckpt_threshold
major_weight = args.major_weight
reg_weight = args.reg_weight
global_maxSimC = 0.0
global_avgSimC = 0.0
global_bmaxSimC = 0.0
global_bavgSimC = 0.0

if not os.path.exists(save_dir):
    print save_dir, 'not exists!!'
    exit()
else:
    if save_dir[-1] != '/':
        print 'please enter a directory for save_dir (path ending with /)'
        exit()

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

avgc_saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
maxc_saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
bavgc_saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
bmaxc_saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

init = tf.global_variables_initializer()

def eval_save_model(sess, t):
    if 'bAvgC' in t:
        saver = bavgc_saver
    elif 'bMaxC' in t:
        saver = bmaxc_saver
    elif 'AvgC' in t:
        saver = avgc_saver
    elif 'MaxC' in t:
        saver = maxc_saver

    save_path = save_dir + "lr_"+str(lr_rate)+"_window_"+str(context_window)+"_batch_"+str(batch_size)+"_sample_"+str(samp_size)+"_sense_"+str(sense_dim)+str(t)+str(ite)
    save_path = saver.save(sess, save_path)
    print "\n========best {} model saved in file: {}========\n".format(t, save_path)

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
                                                        ss.contextual_weight:major_weight,
                                                        ss.reg_weight:reg_weight})

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
    if maxSimC > ckpt_threshold and maxSimC > global_maxSimC:
        global_maxSimC = maxSimC
        eval_save_model(sess, '-bestMaxC-')
    if avgSimC > ckpt_threshold and avgSimC > global_avgSimC:
        global_avgSimC = avgSimC
        eval_save_model(sess, '-bestAvgC-')
    print 'scws AvgSimC =','{:.5f}'.format(avgSimC), 'MaxSimC =','{:.5f}'.format(maxSimC)
    ret_str = 'scws AvgSimC = {:.5f}'.format(avgSimC) + ' MaxSimC = {:.5f}'.format(maxSimC)
    return ret_str, scores, score_name

def train(lang):
    if lang == 'en':
        EOE, input_feed, dynamic_window = en_dh.get_train_batch(batch_size)
        ss = en_ss
        bi_ss = de_ss
        srl = en_srl
        bi_srl = de_srl
    elif lang == 'de':
        EOE, input_feed, dynamic_window = de_dh.get_train_batch(batch_size)
        ss = de_ss
        bi_ss = en_ss
        srl = de_srl
        bi_srl = en_srl
    if EOE:
        return EOE
    
    # select target sense id for target language
    sense_greedy_choice = sess.run(ss.sense_greedy, feed_dict={
                    ss.context_indices:input_feed['context_indices'],
                    ss.bi_info:tf.SparseTensorValue(input_feed['bi_loc'], input_feed['bi_indices'], input_feed['bi_shape']),
                    ss.lengths:input_feed['bi_lengths'],
                    ss.eval_mode:False,
                    ss.contextual_weight:major_weight,
                    ss.reg_weight:reg_weight
                    })
    
    # select target sense id for bilingual language
    bi_sense_greedy_choice = sess.run(bi_ss.sense_greedy, feed_dict={
                    bi_ss.context_indices:input_feed['bi_context_indices'],
                    bi_ss.bi_info:tf.SparseTensorValue(input_feed['bi_loc'], input_feed['bi_indices'], input_feed['bi_shape']),
                    bi_ss.lengths:input_feed['bi_lengths'],
                    bi_ss.eval_mode:True, # no bilingual information
                    bi_ss.contextual_weight:major_weight,
                    bi_ss.reg_weight:reg_weight
                    })
    
    sense_selection_mask = np.random.rand(context_window*2+batch_size) < 0.05
    sense_selected = sense_selection_mask * np.random.randint(0, sense_dim, context_window*2+batch_size) \
      + (1-sense_selection_mask) * sense_greedy_choice

    target_selected, bi_target_selected = data.get_train_sample_batch(batch_size, sense_selected, input_feed, bi_sense_greedy_choice, context_window, sense_dim)
    
    # train target language
    _ = sess.run([srl.train, srl.train_bilingual], feed_dict={
        srl.selected_sense_input_indices:input_feed['selected_sense_input_indices'],
        srl.selected_sense_output_indices:input_feed['selected_sense_output_indices'],
        srl.selected_bi_sense_output_indices:input_feed['bi_selected_sense_input_indices']
        })

    reward = sess.run(srl.reward_sense_prob, feed_dict={
        srl.selected_sense_input_indices:input_feed['selected_sense_input_indices'],
        srl.selected_sense_output_indices:input_feed['selected_sense_output_indices'],
        })
    
    _, cost, ent = sess.run([ss.update, ss.print_cost, ss.print_ent], feed_dict={
        ss.context_indices:input_feed['context_indices'],
        ss.reward_prob:reward,
        ss.target_sense_sampled_indices:target_selected, 
        ss.bi_info:tf.SparseTensorValue(input_feed['bi_loc'], input_feed['bi_indices'], input_feed['bi_shape']),
        ss.lengths:input_feed['bi_lengths'],
        ss.eval_mode:False,
        ss.reg_weight:reg_weight
        })
    
    bi_reward = sess.run(srl.bi_reward_sense_prob, feed_dict={
        srl.selected_sense_input_indices:input_feed['selected_sense_input_indices'],
        srl.selected_bi_sense_output_indices:input_feed['bi_selected_sense_input_indices']
        })
    
    _ = sess.run(bi_ss.update, feed_dict={
        bi_ss.context_indices:input_feed['bi_context_indices'],
        bi_ss.reward_prob:bi_reward,
        bi_ss.bi_info:tf.SparseTensorValue(input_feed['bi_loc'], input_feed['bi_indices'], input_feed['bi_shape']),
        bi_ss.target_sense_sampled_indices:bi_target_selected, 
        bi_ss.lengths:input_feed['bi_lengths'],
        bi_ss.eval_mode:True,
        bi_ss.reg_weight:reg_weight
        })
    
    return EOE

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.graph.finalize()
    sess.run(init)
    
    ite = 0
    with open(log_path, 'w') as fp:
        fp.write('')
    for epoch in range(training_epochs):
        with open(log_path,'a') as fp:
            print_str, scores, score_name = evaluate_scws(sess, en_dh, batch_size, en_ss, en_srl)
            fp.write('completed epoch ')
            epoch_string = '%05d' % (epoch)
            outputString = []
            outputString.append(epoch_string)
            for s in scores:
                outputString.append('{:.4f}'.format(s))

            fp.write(' '.join(outputString)+'\n')
        while True:
            if ite*batch_size % (500*batch_size) == 0:
                print ('-'*89)
                print_str2, scores, score_name = evaluate_scws(sess, en_dh, batch_size, en_ss, en_srl)
                print ('-'*89)
                with open(log_path,'a') as fp:
                    fp.write(print_str2+'\n')
            # train english
            EOE = train('en')
            # train german
            train('de')
            ite += 1
            if EOE:
                break

        print "\n===Epoch:", '%04d' % (epoch+1)

