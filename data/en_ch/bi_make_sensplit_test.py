import sys
import re
import time
import jieba
from nltk.tokenize import sent_tokenize
import string
from opencc import OpenCC
from zhon import hanzi
openCC = OpenCC('t2s')
openCC_final = OpenCC('s2t')

ch_vocab = open('ch_vocab', encoding='utf-8').read().splitlines()
ch_word2id = {}
for line in ch_vocab:
    line = line.split(' ')
    word = line[0]
    ID = line[1]
    ch_word2id[word] = ID

en_vocab = open('en_vocab', encoding='utf-8').read().splitlines()
en_word2id = {}
for line in en_vocab:
    line = line.split(' ')
    word = line[0]
    ID = line[1]
    en_word2id[word] = ID

dataset = open('bcws.txt', encoding='utf-8').read().splitlines()
en_sents = []
ch_sents = []
scores = []
startIdx = list(range(0, len(dataset)//4, 1))
for idx in startIdx:
    ch_sent = dataset[idx*4+1]
    en_sent = dataset[idx*4+2]
    ch_sents.append(ch_sent)
    en_sents.append(en_sent)
    scores.append(float(dataset[idx*4+3].split()[-1]))

out = open('bi_ratings.txt', 'w')

cnt = 0
ch_pos = []
ch_id_sents = []
en_pos = []
en_id_sents = []
for en_sent, ch_sent, score in zip(en_sents, ch_sents, scores):
    # first ch part
    pos1 = ch_sent.find('<')
    pos2 = ch_sent.find('>')
    temp_sent = []
    temp_sent += (jieba.cut(openCC.convert(ch_sent[:pos1]), cut_all=False))
    test_pos = len(temp_sent)
    test_word = re.findall('<(.*?)>', ch_sent)
    if test_word[0] not in ch_word2id:
        continue
    en_test_word = re.findall('<(.*?)>', en_sent)
    if en_test_word[0] not in en_word2id:
        continue
    temp_sent += test_word
    temp_sent += jieba.cut(openCC.convert(ch_sent[pos2+1:]), cut_all=False)
    # word2id
    temp_id = []
    for ch_word in temp_sent:
        try:
            temp_id.append(ch_word2id[openCC_final.convert(ch_word)])
        except:
            temp_id.append(ch_word2id['_UNK'])
    ch_pos.append(test_pos)
    assert temp_sent[test_pos] == test_word[0]
    ch_id_sents.append(temp_id)

    # then en part
    en_sent = en_sent.split(' ')
    temp_sent = []
    for idx, en_word in enumerate(en_sent):
        if '<' in en_word: # target word
            test_pos = idx
            en_word = re.findall('<(.*?)>', en_word)[0]
        try:
            temp_sent.append(en_word2id[en_word])
        except:
            temp_sent.append(en_word2id['_UNK'])
    en_pos.append(test_pos)
    assert temp_sent[test_pos] == en_word2id[en_test_word[0]]
    en_id_sents.append(temp_sent)
    cnt += 1
    out.write('{}\n{}\n{}\n{}\n{}\n'.format(score, ch_pos[-1], ' '.join(ch_id_sents[-1]), en_pos[-1], ' '.join(en_id_sents[-1])))
out.close()
