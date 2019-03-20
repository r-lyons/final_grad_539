"""
Author: Rianne Lyons
Date last modified: 10 December 2017
Filename: final.py
Attempted implementation of the CorEx algorithm.
"""

"""
Downloads training data from the 20 newsgroups dataset using scikit learn, and 
formats the data similarly to as described in Appendix C. Selects lowercase 
alphabetic tokens, calculates their frequency, and associates them with documents. 
Takes the top 10000 most frequent words and gives them a signal according to their 
frequency and keeps them associated with documents.
Requires scikit learn and nltk.
Returns: dw_signal is a dictionary of the form {document index: [(word, signal),...]}.
"""
def process_train_data():
	from sklearn.datasets import fetch_20newsgroups
	import nltk
	dw_signal = {} #{doc:[(word, signal)]}
	word_doc_freq = {} #{word:[[docs], freq]}
	wf_list = []
	print 'getting newsgroups'
	traindt = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.sport.hockey', 'comp.sys.ibm.pc.hardware', 'misc.forsale'], remove=('headers', 'footers', 'quotes'))
	sum_freq = 0
	for i in range(len(traindt.data)): #i is index of document
		doc_tokens = nltk.word_tokenize(traindt.data[i].encode('utf-8'))
		print 'getting words & frequencies for doc', i
		for t in doc_tokens:
			if t.isalpha:
				try:
					word_doc_freq[t.lower()][1] += 1
					sum_freq += 1
					if i not in word_doc_freq[t.lower()][0]:
						word_doc_freq[t.lower()][0].append(i)
				except KeyError:
					word_doc_freq[t.lower()] = [[i], 1]
					sum_freq += 1
					
	top10000 = [(k, val) for k, val in sorted(zip(word_doc_freq.keys(), word_doc_freq.values()), key=lambda wf: wf[1][1])[-10000:]] #lower -> higher
	print 'found top 10000 words'
	avg_freq = float(sum_freq) / float(len(word_doc_freq.keys()))
	for r in range(len(top10000)):
		print 'building signals', r
		dlist = top10000[r][1][0]
		w = top10000[r][0]
		f = float(top10000[r][1][1])
		sig = 0
		if r > 9000:
			for d in range(len(traindt.data)):
				if d in dlist: #word present in document
					if f < avg_freq:
						sig = 1
					else:
						sig = 2
				try:
					dw_signal[d].append((w, sig))
				except KeyError:
					dw_signal[d] = [(w, sig)]
		else:
			for d in range(len(traindt.data)):
				if d in dlist:
					sig = 1
				try:
					dw_signal[d].append((w, sig))
				except KeyError:
					dw_signal[d] = [(w, sig)]	
							
	return dw_signal
	
"""
Converts given documents, words, and their signals into random variables per document.
Parameters: dict_samples is a dictionary of document: [(word, signal)]
Returns: samples_mat is a matrix of document samples containing random variables (tokens), items is the list formed by calling .items() on the parameter.
"""	
def convert_to_drvs(dict_samples):
	from scipy import stats
	import random as rd
	import numpy as np
	samples_mat = []
	temp_sample = []

	items = dict_samples.items()
	print len(items)
	for s in items:
		print items.index(s)
		del temp_sample[:]
		c = int(1+rd.random()*11)
		erasure_prob = 1.0 - (2/c)
		xk = np.array(s[1])
		for n in s[1]:
			x = np.array([n, 0.0])
			if n == 0 or n == 2:
				p = np.array([1.0-erasure_prob, erasure_prob])
			else:
				p = np.array([erasure_prob, 1.0-erasure_prob])
			temp_sample.append(stats.rv_discrete(values=(x, p)))
		samples_mat.append(temp_sample)
	return samples_mat, items

"""
Implementation of the Kronecker delta function.
Parameters: two numeric arguments
Returns: 1 if the arguments are equal, 0 otherwise
"""
def kdelta(arg1, arg2):
	import numpy as np
	if arg1 == arg2:
		return 1
	else:
		return 0

"""
Implements Algorithm 1 and prints the output.
Parameters: samples is a matrix of samples of length n (the number of random variables), sample_items is a list of tuples of documents paired with their words and signals.
"""
def corex_alg(samples, sample_items):
	import random as rd
	from scipy import stats
	import math
	import numpy as np
	k = 2 #Appendix C
	m = 100 #Appendix C
	ld = 0.3
	b = 8
	ymarginals = []
	condmarginals = []
	prod = []
	y_idxs = []
	labels = [rd.random() for N in range(len(samples))]
	y_idxs = [rd.random() for N in range(len(samples))]
	alpha = stats.uniform.rvs(1/2, 1) #t = 0, section 3
	
	
	for t in range(100):
		Zj = 0.000000001 #initialize to small number for log
		Dj = 500*(-math.log(Zj))
		ymarginal = 0.0
		condmarginal = 0.0
		start = 0
		end = len(samples)
		for s in samples[start:end]:
			del ymarginals[:]
			del condmarginals[:]
			del prod[:]
			ymarginal = 0.0
			condmarginal = 0.0
			prev_Dj = Dj
			for n in range(len(s)):
				for N in range(len(samples)): #Appendix B
					if N < k:
						Zj += sample_items[samples.index(s)][1][n][1]
					ymarginal += labels[start+N]
					kd = kdelta(labels[start+N], sample_items[samples.index(s)][1][n][1])
					condmarginal += labels[start+N]*(float(kd)/(float(s[n].pmf(n))+0.01)) #add for normalization
				
				ymarginals.append((1.0/float(len(samples)))*ymarginal)	
				condmarginals.append((1.0/float(len(samples)))*condmarginal)
				prod.append(float(condmarginals[n])/float(ymarginals[n])) #use in eq. 7
			mutinfo = stats.entropy(condmarginals) + stats.entropy(ymarginals) - stats.entropy(condmarginals, ymarginals)
			Dj = 500*(-math.log(Zj))
			if t < 100:
				gamma = 1.0/float(mutinfo)
			else:
				gamma = (1.0+Dj)/float(mutinfo) 
			print 'updating alpha'
			pr = np.array(prod)
			prob_product = np.prod(pr)**(math.exp(gamma))
			labels[samples.index(s)] = (1/Zj)*max(ymarginals)*prob_product
			y_idxs[samples.index(s)] = ymarginals.index(max(ymarginals)) 		
			
	layer1 = []		
	for i in range(len(labels)):
		layer1.append(sample_items[i][1][int(y_idxs[i])][0])
	print layer1
		
	return

"""
Main function that controls processing the data, turning it into random  variables, and running the algorithm on it.
"""
def main():
	dtrain = process_train_data()
	s, i = convert_to_drvs(dtrain)
	corex_alg(s, i)
	
	return
	
if __name__ == '__main__':
	main()
	
	
	
	
	
	
	
	
	
	
	
