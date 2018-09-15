import sys
import string
import re
from collections import defaultdict

def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def clean_data():
    raw_data_en = open(sys.argv[1], encoding='utf-8')
    raw_data_ch = open(sys.argv[2], encoding='utf-8')
    trantab = str.maketrans('','',string.punctuation)
    ret_en = []
    ret_ch = []
    for line_en, line_ch in zip(raw_data_en, raw_data_ch):
        line_en = line_en.strip('\n')
        line_ch = line_ch.strip('\n')
        line_en = line_en.translate(trantab)
        line_ch = line_ch.translate(trantab)
        if len(line_en.strip(' ')) == 0 or len(line_ch.strip(' ')) == 0:
            continue
        line_en = line_en.lower()
        line_ch = line_ch.lower()
        words_en = line_en.split(' ')
        words_ch = line_ch.split(' ')
        line_en = []
        line_ch = []
        for idx, word in enumerate(words_en):
            if is_number(word):
                line_en.append('0')
            elif word.isalnum():
                line_en.append(word)
        if len(line_en) == 0:
            continue
        for idx, word in enumerate(words_ch):
            if is_number(word):
                line_ch.append('0')
            elif word.isalnum():
                line_ch.append(word)
        if len(line_ch) == 0:
            continue
        ret_ch.append(line_ch)
        ret_en.append(line_en)
    
    return ret_en, ret_ch

def build_vocab(data, l, vocab_size):
    word2freq = defaultdict(int)
    word2id = {}
    for line in data:
        for word in line:
            word2freq[word] += 1
    s = [(k, word2freq[k]) for k in sorted(word2freq, key=word2freq.get, reverse=True)]
    # write pad, unk ... to vocab files
    vocab = open(l+"_vocab", 'w', encoding='utf-8')
    vocab.write('_PAD 0 0\n')
    vocab.write('_UNK 1 0\n')
    word2id['_PAD'] = 0
    word2id['_UNK'] = 1
    for idx, t in enumerate(s[:vocab_size]):
        word = t[0]
        freq = t[1]
        word2id[word] = len(word2id)
        vocab.write('{} {} {}\n'.format(word, word2id[word], freq))
    vocab.close()
    out = open(l+'.out', 'w')
    for line in data:
        temp = [str(word2id[word]) if word in word2id else str(word2id['_UNK']) for word in line]
        out.write(' '.join(temp)+'\n')
    out.close()

if __name__ == '__main__':
    data_en, data_ch = clean_data()
    build_vocab(data_en, 'en', int(sys.argv[3]))
    build_vocab(data_ch, 'ch', int(sys.argv[4]))
