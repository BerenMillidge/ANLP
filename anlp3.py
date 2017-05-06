
from __future__ import division
from math import log,sqrt
from collections import defaultdict
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import collections
import matplotlib.pyplot as plt

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a
  username (starts with @).  If so, returns the word unchanged.
  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word
  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
  '''Compute the pointwise mutual information using cooccurrence counts.
  :type c_xy: int
  :type c_x: int
  :type c_y: int
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value
  '''
  return log(c_xy * N / ((c_x) * (c_y)), 2)

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print "Warning: PMI is incorrectly defined"
else:
    print "PMI check passed"

# def tanimoto(v0, v1):
#     dotprod = 0
#
#     for x in v0.keys():
#         v0[x] = 1
#
#     for x in v1.keys():
#         v1[x] = 1
#
#     for ind in v0.keys():
#         if v1.get(ind):
#             dotprod += v0[ind] * v1[ind]
#     sim = dotprod / (veclen(v0) + veclen(v1) - dotprod)
#     return sim


def jaccard(v0, v1):

    numerator = 0
    denominator = 0
    for ind in v0.keys():
        if v1.get(ind):
            numerator += min(v0[ind], v1[ind])
            denominator += max(v0[ind], v1[ind])
    return numerator / denominator

def dice(v0, v1):

    numerator = 0
    for ind in v0.keys():
        if v1.get(ind):
            numerator += min(v0[ind], v1[ind])
    denominator = sum(v0.values()) + sum(v1.values())
    return (2 * numerator) / denominator

def js(v0, v1):
    v0 = defaultdict(int, v0)
    v1 = defaultdict(int, v1)
    avg = {}
    words = set(v0).union(v1)
    for ind in words:
            avg[ind] = (v0[ind] + v1[ind]) / 2
    v0sum = 0
    v1sum = 0
    for x in v0:
        if v0[x] != 0:
            v0sum += v0[x] * log((v0[x] / avg[x]), 2)
    for x in v1:
        if v1[x] != 0:
            v1sum += v1[x] * log((v1[x] / avg[x]), 2)
    return (v0sum + v1sum) / 2

def cos_sim(v0,v1):
    '''Compute the cosine similarity between two sparse vectors.
    :type v0: dict
    :type v1: dict
    :param v0: first sparse vector
    :param v1: second sparse vector
    :rtype: float
    :return: cosine between v0 and v1
    '''

    dotprod = 0
    for ind in v0.keys():
        if v1.get(ind):
            dotprod += v0[ind] * v1[ind]

    cos = dotprod / (veclen(v0) * veclen(v1))

    return cos
    # We recommend that you store the sparse vectors as dictionaries
    # with keys giving the indices of the non-zero entries, and values
    # giving the values at those dimensions.

    #You will need to replace with the real function
    # print "Warning: cos_sim is incorrectly defined"
    # return 0


def veclen(vec, sq = False):
    '''Calculate vector lenght
    :type vec: dict
    '''
    sqsum = 0
    for value in vec.values():
        sqsum += value ** 2
    if sq:
        return sqsum
    else:
        return sqrt(sqsum)


def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for all words, using PPMI.
    These should be sparse vectors.
    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''

    vectors = {}
    for wid0 in wids:
        ##you will need to change this
        vectors[wid0] = {}
        for coword, conum in co_counts[wid0].items():
            pmi = PMI(conum, o_counts[wid0], o_counts[coword], tot_count)
            if pmi > 0:
                vectors[wid0][coword] = pmi
    return vectors

def create_prob_vectors(wids, o_counts, co_counts):
    '''Creates context vectors for all words, using probability.
    These should be sparse vectors.
    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''

    vectors = {}
    for wid0 in wids:
        vectors[wid0] = {}
        for coword, conum in co_counts[wid0].items():
            vectors[wid0][coword] = conum / o_counts[wid0]

    return vectors

def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.
  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(fp.next())
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.
  :type similarities: dict
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print "%0.2f\t%-30s\t%d\t%d" % (similarities[pair],word_pair,o_counts[pair[0]],o_counts[pair[1]])


def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.
  :type items: list
  :param items: the list to pair up
  :return: list of pairs
  '''
  return [(x, y) for x in items for y in items if x < y]


fp = open("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/wid_word");
wid2word={}
word2wid={}
for line in fp:
    widstr,word=line.rstrip().split("\t")
    wid=int(widstr)
    wid2word[wid]=word
    word2wid[word]=wid

test_words = ["cat", "dog", "mouse", "computer","@justinbieber","san", "francisco", "love", "hate", "pizza", "terrorist", "tweet", "twitter", "@ladygaga","@arianagrande","lmao","lol"]
stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs
wid_pairs = make_pairs(all_wids)


#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

#make the word vectors
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
probvectors = create_prob_vectors(all_wids,o_counts, co_counts, )

# compute cosine similarites for all pairs we consider
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
j_sims = {(wid0,wid1): jaccard(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
d_sims = {(wid0,wid1): dice(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
js_sims = {(wid0,wid1): js(probvectors[wid0],probvectors[wid1]) for (wid0,wid1) in wid_pairs}

#print "Sort by cosine similarity"
#print_sorted_pairs(c_sims, o_counts)

#print "Sort by Jaccard similarity"
#print_sorted_pairs(j_sims, o_counts)

#print "Sort by Jaccard-Curran similarity"
#print_sorted_pairs(d_sims, o_counts)

#print "Sort by Jensen-Shannon divergence"
#print_sorted_pairs(js_sims, o_counts)

#results = [c_sims, j_sims, d_sims, js_sims]
#name = ["Cosine similarity",  "Jaccard similarity", "Jaccard-Curran similarity", "Jensen-Shannon divergence"]

def replace_keys(orddict):
	newdict = collections.OrderedDict()
	i = len(orddict.keys())
	for key in orddict.keys():
		newdict[i] = orddict[key]
		i-=1
	return newdict

def normalise_vals(array):
	column = array.T[1]
	row = array.T[0]
	total = np.sum(column)
	els = []
	for element in column:
		element = element/total
		els.append(element)
	els = np.array(els)
	arr = np.column_stack([row,els])
	return arr

def sumcolumn(array):
	col = array.T[1]
	return np.sum(col)
		

def get_ordered_norm(x):
	x_ord = collections.OrderedDict(x)
	#y_ord = collections.OrderedDict(y)
	x_ord = replace_keys(x_ord)
	#y_ord = replace_keys(y_ord)
	xsimarray = np.array(sorted(x_ord.items(),key= lambda x:x[1],reverse=True))
	#ysimarray = np.array(sorted(y_ord.items(),key = lambda x:x[1],reverse=True))
	normx = normalise_vals(xsimarray)
	#normy = normalise_vals(ysimarray)
	return normx#,normy

def get_freqsum(x):
	x_ord = collections.OrderedDict(x)
	els = []
	for wids in x_ord.keys():
		freq1 = o_counts[wids[0]]
		freq2 = o_counts[wids[1]]
		els.append(freq1+freq2)
	x_ord = replace_keys(x_ord)
	i = 0
	for key in x_ord.keys():
		x_ord[key] = els[i]
		i +=1
	arr = np.array(sorted(x_ord.items(),key= lambda x:x[1],reverse=True))
	return arr

def plot_stuff(x,y):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(x.T[0])
	ax.plot(y.T[0])
	plt.show()

def truncate_array(array,n):
	splits = np.split(array,[0,n])
	print splits[1]
	return splits[1]
	


def compute_spearmans(x,y):
	#normx,normy = get_ordered_normed(x,y)
	plot_stuff(x,y)
	return scipy.stats.spearmanr(x.T[0],y.T[0])

def compute_pearsons(x,y):
	normx, normy = get_ordered_normed(x,y)
	return scipy.stats.pearsonr(normx.T[1],normy.T[1])

fc_sims = get_freqsum(c_sims)
fj_sims = get_freqsum(j_sims)
fd_sims = get_freqsum(d_sims)
fjs_sims = get_freqsum(js_sims)

c_sims = get_ordered_norm(c_sims)
j_sims = get_ordered_norm(j_sims)
d_sims = get_ordered_norm(d_sims)
js_sims =get_ordered_norm(js_sims)


#print fc_sims

print compute_spearmans(fc_sims,c_sims)
#print compute_pearsons(d_sims,j_sims)






"""
for i in range(0,4):
    dic = results[i]

    print name[i]
    keys = np.array(dic.keys())
    vals = np.array(dic.values())
    list_keys, key_index = np.unique(keys, return_inverse=True)
    key_index = key_index.reshape(-1, 2)
    n = len(list_keys)
    mtx = np.zeros((n, n) ,dtype=vals.dtype)
    mtx[key_index[:,0], key_index[:,1]] = vals
    mtx += mtx.T
    df = pd.DataFrame(mtx)
    wds = [wid2word[k] for k in list_keys]
    df.columns = wds
    df.index = wds
    print df
    if 'dfo' not in globals():
        dfo = pd.DataFrame(columns=(wds))
    dfo = dfo.append(df)

words = '_'.join([wid2word[k] for k in list_keys])
n = words + '.csv'
dfo.to_csv(n)"""
