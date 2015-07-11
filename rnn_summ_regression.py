from rnn_regression import rnn
from xmlparser_multiple import parse_bc3
import numpy
from random import shuffle
from evaluation import divide_data,weightRecall
from python_rouge import rouge

def sorted_index(myList):
        return [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1])]
class rnn_summ(object):
	def __init__(self,train,test,write_file = None):
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
                self.write_file = write_file
		'''
		if self.write_file is not None:
			self.file = open(self.write_file,'w')
		'''
		self.train_epoch = 0
	def shuffle(self):
		shuffle(self.mails)
	def init_rnn(self,L1,L2):
		if self.write_file is not None:
			self.rnn_model = rnn(70,self.length,1,L1,L2,True)
		else:
			self.rnn_model = rnn(70,self.length,1,L1,L2)
	def close_file(self):
		if self.file is not None:
			self.file.close()
	def rnn_train(self,learning_rate):
		all_error = 0
		self.train_epoch += 1
		if self.write_file is not None and self.train_epoch %600 == 0:
			f = open(self.write_file + "/epoch_" + str(self.train_epoch),'w')
			#self.file.write("train_epoch : "+str(self.train_epoch)+'\n')
		for i in self.train:
			for j in i.thread:
				input_ins = []
				label_ins = []
				for k in j.sentences:
					'''	
					if k.score == 0:
						label_ins.append([0.0])
					elif k.score >0 and k.score < 0.35:
						label_ins.append([1.0])
					elif k.score >0.35 and k.score < 0.7:
                                                label_ins.append([2.0])
					else:
						label_ins.append([3.0])
					'''
					label_ins.append([float(k.score)])
					input_ins.append(k.feature)
				label_array = label_ins
				input_ins = input_ins + input_ins
				label_ins = label_ins + label_ins
				input_ins = numpy.asarray(numpy.float32(input_ins))
                		label_ins = numpy.asarray(numpy.float32(label_ins))
				self.rnn_model.flush_hidden()
				error = self.rnn_model.sentence_train(input_ins,label_ins,learning_rate)
				if self.train_epoch % 600 == 0 and self.write_file is not None:
					self.rnn_model.get_hidden(input_ins)
					if self.write_file is not None:
						write_string = ""
						for i,j in zip(self.rnn_model.hidden_layer,label_array):
							for k in i:
								write_string += (str(k) +"\t")
							write_string += ('\t' + str(j[0]) + '\n')
						f.write(write_string)
          			all_error += error
		if self.write_file is not None and self.train_epoch %600 == 0:
			f.close()
		return all_error
	def rnn_test(self):
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
					score = i#(i[1] * 0.33) + (i[2] * 0.66) + (i[3] * 1)
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
		score =  weightRecall(self.test,produce_set)
                print score
		rouge_eval = rouge(self.test,produce_set)
                rouge_score =  rouge_eval.eval()['rouge_l_f_score']
                print rouge_score
		return score,rouge_score
'''
corpus = 'bc3/bc3corpus.1.0/corpus.xml'
annotation = 'bc3/bc3corpus.1.0/annotation.xml'
mails = parse_bc3(corpus,annotation)
tmp = rnn_summ(mails[0:36],mails[36:])
avg_score = 0
avg_rouge = 0
for i in range(1):
	tmp.init_rnn(0.01,0)
	#tmp.shuffle()
	rate = 0.2
	for j in range(2500):
		#error = tmp.rnn_train(0.2)
		if j %100 == 0:
			rate = rate*0.9
		error = tmp.rnn_train(rate)
		if j % 50 == 0:
			print error
	score,rouge_score = tmp.rnn_test()
	avg_score += score
	avg_rouge += rouge_score
print "score:  " + str(avg_score)
print "rouge:  " + str(avg_rouge) 
'''
