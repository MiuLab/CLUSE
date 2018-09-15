import numpy as np
from random import random, randint
from utils import PAD_ID
import sys

class data_helper(object):
    def __init__(self, path, sense_dim, batch_size, context_window, lang):
        self.lang = lang
        self.path = path
        self.sense_dim = sense_dim
        self.data = [PAD_ID for i in xrange(context_window*2)]
        self.bi_context = [[PAD_ID] for i in xrange(context_window*2)]
        self.leftBoundary = [0 for i in xrange(context_window*2)]
        self.rightBoundary = [0 for i in xrange(context_window*2)]
        self.rnd_index = 0
        self.context_window = context_window
        self.gen = self.file_gen()
        self.load_vocab(path, sense_dim, batch_size)
        if self.lang == 'en':
            self.read_scws(path)
    
    def read_scws(self, path):
        context_window = self.context_window
        test_score = []
        testLocation = []
        testData = []
        with open(path+'en_test.txt') as fp:
            for e, line in enumerate(fp):
                line = line.strip()
                if e % 5 == 0:
                    test_score.append(float(line))
                elif e % 5 == 1 or e % 5 == 3:
                    testLocation.append(int(line))
                else:
                    testData.append(map(int,line.split(' ')))
            
        for i in xrange(len(testData)):
            original_word = testData[i][testLocation[i]]
            original_seq = testData[i]
            if testLocation[i] - context_window < 0:
                testData[i] = [PAD_ID for t in xrange(context_window - testLocation[i])]+testData[i]
            elif testLocation[i] - context_window > 0:
                testData[i] = testData[i][testLocation[i]-context_window:]
            if len(testData[i])>2*context_window+1:
                testData[i] = testData[i][:2*context_window+1]
            elif len(testData[i])<2*context_window+1:
                testData[i] += [PAD_ID for t in xrange(2*context_window + 1 - len(testData[i]))]

            testLocation[i] = context_window
            assert(testData[i][testLocation[i]] == original_word)
            assert(len(testData[i]) == 2*context_window + 1)

        testData = np.asarray(testData)
        self.scws_testData = testData
        self.scws_testLocation = testLocation
        self.scws_test_score = test_score

    def load_vocab(self, path, sense_dim, batch_size):
        #id2count = [0 for i in xrange(99160)]
        # make sure that ID is in ascending sorted order
        id2count = []
        self.id2senseNum = []
        self.word2id = {}
        self.id2word = {}
        subsampling_factor = 1e-4
        vocab_size = 0
        sense_size = 0
        with open(path+self.lang+'_vocab') as fp:
            for line in fp:
                line = line.strip().split(' ')
                word = line[0]
                ID = int(line[1])
                self.word2id[word] = ID
                self.id2word[ID] = word
                freq = float(line[2])
                id2count.append(freq)
                self.id2senseNum.append(sense_dim)
                vocab_size += 1
                sense_size += sense_dim
        self.vocab_size = vocab_size
        self.sense_size = sense_size
        self.id2count = np.asarray(id2count)
        self.id2freq = self.id2count / np.sum(self.id2count)

        self.rnd = []
        for i in xrange(batch_size):
            self.rnd.append(random())
        for i in xrange(len(self.id2freq)):
            if i <= 3:
                continue
            if self.id2freq[i] == 0:
                continue
            self.id2freq[i] = (np.sqrt(self.id2freq[i]/subsampling_factor)+1)*(subsampling_factor/self.id2freq[i])

        self.sense_counts = [0 for i in xrange(sense_size)]
        for i in xrange(vocab_size):
            for k in xrange(self.id2senseNum[i]):
                self.sense_counts[i*sense_dim+k] = self.id2count[i]/self.id2senseNum[i]

    def file_gen(self):
        while True:
            f = open(self.path+self.lang+'.out')
            for line in f:
                yield 0, line
            yield 1, ''
            f.close()

    def set_bi_gen(self, bi_gen):
        # set bilingual data helper to generate bilingual context
        self.bi_gen = bi_gen

    def get_train_batch(self, batch_size):
        context_window = self.context_window
        data = self.data[:batch_size+context_window*4]
        bi_context = self.bi_context[:batch_size+context_window*4]
        leftBoundary = self.leftBoundary[:batch_size+context_window*4]
        rightBoundary = self.rightBoundary[:batch_size+context_window*4]
        sense_dim = self.sense_dim
        # construct a batch of training data
        while len(data) < batch_size + context_window*4:
            EOE, line = self.gen.next()
            bi_EOE, bi_line = self.bi_gen.next()
            if EOE:
                return 1, '', ''
            try:
                line = map(int, line.strip().split(' '))
                bi_line = map(int, bi_line.strip().split(' '))
                if len(bi_line) > 20:
                    sampled_idx = np.sort(np.random.permutation(range(len(bi_line)))[:20])
                    bi_line = [bi_line[i] for i in sampled_idx]
            except:
                continue
            if len(line) == 0 or len(bi_line) == 0:
                continue
            # subsampling
            offset = len(data)
            counter = 0
            for i in xrange(len(line)):
                self.rnd_index = (self.rnd_index+1)%len(self.rnd)
                if self.id2freq[line[i]] < self.rnd[self.rnd_index]:
                    continue
                data.append(line[i])
                bi_context.append(bi_line) # for every word in target language, it has the same bilingual context, namely, the whole bilingual sentence
                leftBoundary.append(min(context_window, counter))
                counter+=1
            lineLength = len(data) - offset
            for i in xrange(lineLength):
                rightBoundary.append(min(context_window, lineLength-1-i))
            if lineLength == 1:
                data = data[:-1]
                bi_context = bi_context[:-1]
                leftBoundary = leftBoundary[:-1]
                rightBoundary = rightBoundary[:-1]

            assert(len(data) == len(leftBoundary))
            assert(len(data) == len(rightBoundary))
            assert(len(data) == len(bi_context))

        c_indices = [None] * (context_window*2+batch_size)
        bi_c_indices = []
        bi_indices = []
        bi_loc = []
        max_col = 0
        indices = range(context_window, context_window*3+batch_size)
        dynamic_window = np.random.randint(1, context_window+1, len(indices))
        bi_lengths = []
        for i in xrange(len(indices)):
            # get the contextual word indices for every target word
            leftBoundary[indices[i]] = min(leftBoundary[indices[i]], dynamic_window[i])
            rightBoundary[indices[i]] = min(rightBoundary[indices[i]], dynamic_window[i])
            c_indices[i] = [PAD_ID]*(context_window-leftBoundary[indices[i]]) \
                + data[indices[i]-leftBoundary[indices[i]]:indices[i]+1+rightBoundary[indices[i]]] \
                + [PAD_ID] * (context_window-rightBoundary[indices[i]])
            
            # moreover, select the sense of bilingual language at the same time
            bi_len = len(bi_context[indices[i]])
            try:
                bi_idx = np.random.choice(bi_len)
            except:
                print (bi_context[indices[i]])
                exit(1)
            # get the context word id of selected bi_idx
            bi_left = [PAD_ID]*(context_window-min(bi_idx, context_window)) + bi_context[indices[i]][max(0, bi_idx-context_window):bi_idx]
            bi_right = bi_context[indices[i]][bi_idx+1:min(bi_idx+context_window+1, bi_len)] + [PAD_ID]*max(context_window-(bi_len-bi_idx-1), 0)
            bi_c_indices.append(bi_left + [bi_context[indices[i]][bi_idx]] + bi_right)
            bi_indices += bi_context[indices[i]]
            for j in range(len(bi_context[indices[i]])):
                bi_loc.append([i, j])
            max_col = max(max_col, len(bi_context[indices[i]]))
            bi_lengths.append(len(bi_context[indices[i]]))
        
        input_feed = {'context_indices': c_indices, 'bi_indices': bi_indices, 'bi_shape': [context_window*2+batch_size, max_col], 'bi_loc': bi_loc, 'bi_context_indices':bi_c_indices, 'bi_lengths':bi_lengths}
        data = data[batch_size:]
        bi_context = bi_context[batch_size:]
        leftBoundary = leftBoundary[batch_size:]
        rightBoundary = rightBoundary[batch_size:]

        return 0, input_feed, dynamic_window

def get_train_sample_batch(batch_size, sense_selected, input_feed, bi_sense_greedy_choice, context_window, sense_dim):
    # [context_window*2+batch_size]
    c_indices = input_feed['context_indices']
    bi_c_indices = input_feed['bi_context_indices']
    collocation_selected = []
    target_selected = []
    bi_target_selected = []
    selected_s_input_indices = []
    bi_selected_s_input_indices = []
    selected_s_output_indices = []

    for i in xrange(context_window, batch_size+context_window):
        # get the target word indices
        index = c_indices[i][context_window]*sense_dim + sense_selected[i]
        target_selected.append(i*sense_dim+sense_selected[i])
        bi_target_selected.append(i*sense_dim+bi_sense_greedy_choice[i])
        bi_index = bi_c_indices[i][context_window]*sense_dim + bi_sense_greedy_choice[i]
        left = context_window*2
        right = 0

        for t in xrange(context_window*2+1):
            if t == context_window or c_indices[i][t] == PAD_ID:
                continue
            else:
                if t < left:
                    left = t
                if t > right:
                    right = t

        selected_s_input_indices.append(index)
        bi_selected_s_input_indices.append(bi_index)
        if left > right:
            print left, right
            print c_indices[i]
        assert(left <= right)
        if left < right:
            t = randint(left+1, right)
        else:
            t = left
        if t == context_window:
            t = left
        t_index = i + t - context_window

        t_index = c_indices[t_index][context_window]*sense_dim+sense_selected[t_index]
        selected_s_output_indices.append(t_index)

    input_feed['selected_sense_input_indices'] = selected_s_input_indices
    input_feed['selected_sense_output_indices'] = selected_s_output_indices
    input_feed['bi_selected_sense_input_indices'] = bi_selected_s_input_indices

    return target_selected, bi_target_selected
