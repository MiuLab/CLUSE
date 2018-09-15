from opencc import OpenCC
from zhon.hanzi import punctuation as c_punc
import re
import sys

openCC = OpenCC('s2t') 
out = open('ch_tra.txt', 'w', encoding='utf-8')
f = open(sys.argv[1], encoding='utf-8').read().splitlines()
for line in f:
    temp = []
    for word in line.split(' '):
        if word not in c_punc:
            converted = openCC.convert(word)
            temp.append(converted)
    out.write(' '.join(temp)+'\n')
out.close()
