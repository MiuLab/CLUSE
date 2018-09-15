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
context_window = args.context_window
sense_dim = args.sense_number
embedding_dim = args.selection_dim
sense_embedding_dim = args.representation_dim
dataset_prefix = args.dataset_dir
ckpt_path = args.ckpt_path

# read in data
en_dh = data.data_helper(dataset_prefix, sense_dim, batch_size, context_window, 'en')
de_dh = data.data_helper(dataset_prefix, sense_dim, batch_size, context_window, 'de')
en_dh.set_bi_gen(de_dh.file_gen())
de_dh.set_bi_gen(en_dh.file_gen())

# construct embedding and model
en_emb = embedding(en_dh.vocab_size, en_dh.sense_size, embedding_dim, sense_embedding_dim, sense_dim)
de_emb = embedding(de_dh.vocab_size, de_dh.sense_size, embedding_dim, sense_embedding_dim, sense_dim)

init = tf.global_variables_initializer()
best_saver = tf.train.Saver(tf.all_variables())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # reload model
    best_saver.restore(sess, ckpt_path)
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
