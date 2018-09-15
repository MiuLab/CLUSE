from datetime import datetime
import gzip, cPickle, pdb, os
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import numpy as np

PAD_ID = 0

def calAvgSimC(test_score, senseVec1, senseScore1,senseVec2, senseScore2):
  assert(len(senseVec1)==len(senseVec2))
  avgCos = []
  for t in xrange(len(senseVec1)):
    thisCos = []
    p1 = (senseScore1[t])
    p2 = (senseScore2[t])
    for i in xrange(len(senseVec1[t])):
      for j in xrange(len(senseVec2[t])):
        thisCos.append((1-cosine(senseVec1[t][i],senseVec2[t][j]))*p1[i]*p2[j])
    avgCos.append(np.sum(thisCos))
  return spearmanr(test_score, avgCos)[0]

def calMaxSimC(test_score, senseVec1, senseScore1,senseVec2, senseScore2):
  assert(len(senseVec1)==len(senseVec2))
  maxCos = []
  for t in xrange(len(senseVec1)):
    i = np.argmax(senseScore1[t])
    j = np.argmax(senseScore2[t])
    thisCos = (1-cosine(senseVec1[t][i],senseVec2[t][j])) 
    maxCos.append(thisCos)
  return spearmanr(test_score, maxCos)[0]

class Timer(object):
	def __init__(self, name=None, verbose=2):
		self.name = name
		self.verbose = verbose;

	def __enter__(self):
		if self.name and self.verbose >= 1:
			print("...", self.name)
		self.start = datetime.now()
		return self

	def __exit__(self, type, value, traceback):
		if self.verbose >= 2:
			if self.name:
				print '...', self.name, "done in", datetime.now() - self.start
			else:
				print "Elapsed:", datetime.now() - self.start
