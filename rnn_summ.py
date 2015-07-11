from rnn import rnn
from xmlparser_multiple import parse_bc3
import numpy
from random import shuffle
from evaluation import divide_data,weightRecall
from python_rouge import rouge

def sorted_index(myList):
        return [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1])]
class rnn_summ(object):
	def __init__(self,train,test):
		f = open('bc3_sentiment_vectors.txt')
		vectors = []
		index = []
		dic = {}
		for i in f.readlines():
			dic[i.split()[0]] = i.split()[1:]
			self.length = len(i.split()[1:])
		self.train = train
		self.test = test
        	for i in (self.train+self.test):
                	for j in i.thread:
                        	for k in j.sentences:
                                	index = '_mail' + str(i.number) + '_' + str(k.index)
                                	#print index
                                	k.feature = dic[index]
                	
	def shuffle(self):
		shuffle(self.mails)
	def init_rnn(self,L1,L2):
		self.rnn_model = rnn(70,self.length,4,L1,L2)
	def rnn_train(self,learning_rate):
		all_error = 0
		for i in self.train:
			for j in i.thread:
				input_ins = []
				label_ins = []
				for k in j.sentences:
					if k.score == 0:
						label_ins.append(0)
					elif k.score >0 and k.score < 0.35:
						label_ins.append(1)
					elif k.score >0.35 and k.score < 0.7:
                                                label_ins.append(2)
					else:
						label_ins.append(3)
					input_ins.append(k.feature)
				input_ins = input_ins + input_ins
				label_ins = label_ins + label_ins
				input_ins = numpy.asarray(numpy.float32(input_ins))
                		label_ins = numpy.asarray(numpy.int32(label_ins))
				error = self.rnn_model.sentence_train(input_ins,label_ins,learning_rate)
          			all_error += error
		return all_error
	def rnn_test(self,write_folder = None):
		produce_set = []
		for i in self.test:
			produce_set.append([])
			score_list = []
			index_list = []
			for j in i.thread:
				input_ins = []
				label_ins = []
				index = []
				for k in j.sentences:
                                	input_ins.append(k.feature)
					index.append(k.index)
                                input_ins = input_ins + input_ins
                                input_ins = numpy.asarray(numpy.float32(input_ins))
                           	softmax_array = self.rnn_model.prob(input_ins)
				count = 0
                		for i in softmax_array[(len(softmax_array)/2) :]:
					score = (i[1] * 0.33) + (i[2] * 0.66) + (i[3] * 1)
					score_list.append(score)
					index_list.append(index[count])
					count += 1
			sorted_index_array = sorted_index(score_list)
                        sen_length = 0
                        for j in range(len(index_list)):
                        	if sen_length < float(len(index_list))*0.3:
                                	produce_set[-1].append(index_list[sorted_index_array[len(index_list)-j-1]])
                                        sen_length += 1
                                else:
                                	break
		score =  weightRecall(self.test,produce_set,write_folder)
                print score
		rouge_eval = rouge(self.test,produce_set)
                rouge_score =  rouge_eval.eval()['rouge_l_f_score']
                print rouge_score
		return score,rouge_score
'''
corpus = 'bc3/bc3corpus.1.0/corpus.xml'
annotation = 'bc3/bc3corpus.1.0/annotation.xml'
mails = parse_bc3(corpus,annotation)
tmp = rnn_summ(mails)
avg_score = 0
avg_rouge = 0
for i in range(5):
	tmp.init_rnn(0.01,0)
	tmp.shuffle()
	for j in range(400):
		error = tmp.rnn_train(0.1)
		if j % 40 == 0:
			print error
	score,rouge_score = tmp.rnn_test()
	avg_score += score
	avg_rouge += rouge_score
print "score:  " + str(avg_score/5.)
print "rouge:  " + str(avg_rouge/5.) 
'''
