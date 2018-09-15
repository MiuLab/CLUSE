python3 convert2tra.py $2
python3 preprocess.py $1 ch_tra.txt $3 $4
python3 bi_make_sensplit_test.py
python2.7 make_sensplit_test.py en_vocab ratings.txt en_test.txt
