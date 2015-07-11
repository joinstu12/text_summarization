from swda import CorpusReader
from rnn import rnn
import nltk
import numpy
from random import shuffle
import cPickle as pickle
class swda_reader(object):
	def __init__(self):
		self.corpus = CorpusReader('swda')
	def transcript_reader(self):
		def is_neg(tag):
			if (tag == 'sv') or (tag == 'sd'):
				return True
			return False
		def is_pos(tag):
			if (tag == 'qy') or (tag == 'qw') or (tag == 'qh'):
				return True
			return False
		pos_data = []
		neg_data = []
		for trans in self.corpus.iter_transcripts():
			pool = []
			for utt in trans.utterances:
				if utt.damsl_act_tag() == '+':
					pool.append(utt)
				else:
					if len(pool) > 0:
						pool.append(utt)
					else:
						pool = [utt]
					if is_neg(utt.damsl_act_tag()):
                                        	neg_data.append(pool)
					elif is_pos(utt.damsl_act_tag()):
						pos_data.append(pool)
					pool = []
				'''
				if is_neg(utt.damsl_act_tag()) or is_pos(utt.damsl_act_tag()):
					print utt.pos_words()
					print utt.damsl_act_tag()
				'''
		return pos_data,neg_data
class pos_based_qa_model(object):
	def __init__(self,data):
		self.pos_data,self.neg_data = data
		print 'number of pos data : ' + str(len(self.pos_data))
		print 'number of neg data : ' + str(len(self.neg_data))
		#self.construct_dic(0.8)
		self.split_data(0.8,shuffle_data = False)
	def get_threshold(self):
		self.pos_model.decision_stump(self.pos_test,self.neg_test[0:len(self.pos_test)])
	def construct_dic(self,unknown_rate):
		self.dic = {}
		for i in self.pos_data[0:int(len(self.pos_data)*unknown_rate)]:
			for utt in i:
				utt.pos_tag = []
				for j in nltk.pos_tag(utt.pos_words()):
					utt.pos_tag.append(j[1])
				for tag in utt.pos_tag:
					if tag not in self.dic:
						self.dic[tag] = len(self.dic)
		for i in self.neg_data[0:int(len(self.neg_data)*unknown_rate)]:
                        for utt in i:
				utt.pos_tag = []
				for j in nltk.pos_tag(utt.pos_words()):
                                        utt.pos_tag.append(j[1])
                                for tag in utt.pos_tag:
					if tag not in self.dic:
                                        	self.dic[tag] = len(self.dic)	
		self.dic['unk'] = len(self.dic)
		print 'dic length : ' + str(len(self.dic))
	def split_data(self,train_rate,shuffle_data = False):
		if shuffle_data == True:
			shuffle(self.pos_data)
			shuffle(self.neg_data)
		self.pos_train = self.pos_data[0:int(len(self.pos_data)*train_rate)]
		self.pos_test = self.pos_data[int(len(self.pos_data)*train_rate):]
		self.neg_train = self.neg_data[0:int(len(self.neg_data)*train_rate)]
                self.neg_test = self.neg_data[int(len(self.neg_data)*train_rate):]
	def init_qa_models(self):
		self.pos_model = tag_transition_model(self.pos_train,self.dic)
		self.pos_model.init_rnn_model(100,0.0,0.0)
		self.neg_model = tag_transition_model(self.neg_train,self.dic)
		self.neg_model.init_rnn_model(100,0.0,0.0)
	def train_qa_models(self):
		print "training pos model ..."
		for i in range(100):
			self.pos_model.train_model(0.001,'pos_model')
		print "training neg model ..."
		self.neg_model.train_model(0.001,'neg_model')
	def save_dic(self,filename):
                file = open(filename,'wb')
                pickle.dump(self.dic, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.close()
        def load_dic(self,filename):
                file = open(filename,'rb')
                self.dic = pickle.load(file)
                file.close()
	def save_model(self,file_pos=None,file_neg=None,dic=None):
		if file_pos is not None:
			self.pos_model.save(file_pos)
		if file_neg is not None:
                        self.neg_model.save(file_neg)
		if dic is not None:
                        self.save_dic(dic)
	def load_model(self,file_pos=None,file_neg=None,dic=None):
		if file_pos is not None:
                        self.pos_model.load(file_pos)
                if file_neg is not None:
                        self.neg_model.load(file_neg)
                if dic is not None:
                        self.load_dic(dic)


class tag_transition_model(object):
	def __init__(self,data,dic):
		self.data = data
		self.dic = dic
	def one_hot_representation(self,tag):
                vector = numpy.zeros(len(self.dic))
                vector[self.dic[tag]] = 1
                return vector
	def init_rnn_model(self,num_hidden,L1,L2):
		self.rnn_model = rnn(num_hidden,len(self.dic),len(self.dic),L1,L2)
	def generate_input(self,data):
		input_ins = []
		label_ins = []
		for utt in data:
			pos_tag = []
                        for j in nltk.pos_tag(utt.pos_words()):
                        	pos_tag.append(j[1])
				#print j[1]
			for tag_index in range(len(pos_tag)):
				if tag_index < (len(pos_tag) -1):
					if pos_tag[tag_index] in self.dic:
						input_ins.append(self.one_hot_representation(pos_tag[tag_index]))
					else:
						input_ins.append(self.one_hot_representation('unk'))
				if tag_index > 0:
					if pos_tag[tag_index] in self.dic:
						label_ins.append(self.dic[pos_tag[tag_index]])
					else:
                                                label_ins.append(self.dic['unk'])
		return input_ins,label_ins
	def generate_sentence_input(self,data):
		input_ins = []
                label_ins = []
                pos_tag = []
               	for j in nltk.pos_tag(data.split()):
                	pos_tag.append(j[1])
               	for tag_index in range(len(pos_tag)):
                	if tag_index < (len(pos_tag) -1):
                        	if pos_tag[tag_index] in self.dic:
                                	input_ins.append(self.one_hot_representation(pos_tag[tag_index]))
                                else:
                                        input_ins.append(self.one_hot_representation('unk'))
                        if tag_index > 0:
                                if pos_tag[tag_index] in self.dic:
                                       	label_ins.append(self.dic[pos_tag[tag_index]])
                                else:
                                        label_ins.append(self.dic['unk'])
                return input_ins,label_ins
	def train_model(self,learning_rate,early_save_file=None,early_save_epoch =1000):
		count = 0.
		error = 0.
		for row in self.data:
			input_ins,label_ins = self.generate_input(row)
			if not label_ins:
                        	continue
                        input_ins = numpy.asarray(numpy.int32(input_ins))
                        label_ins = numpy.asarray(numpy.int32(label_ins))
                        error += self.rnn_model.sentence_train(input_ins,label_ins,learning_rate)
			count += 1.
			if count %1000 == 0:
				print "epoch" + str(count/1000) + str(error/1000)
				error = 0.
			if (early_save_file is not None) and (count % early_save_epoch == 0):
				self.save(early_save_file)
				
	def save(self,filename):
		self.rnn_model.save(filename)
	def load(self,filename):
		self.rnn_model.load(filename)
	def baseline(self,data):
		if data[-1].pos_words()[-1] == '?':
			return 1
		else:
			return 0
	def baseline_sentence(self,data):
		if data.split()[-1] == '?':
			return 1
		else:
			return 0
	def decision_stump(self,pos_valid_data,neg_valid_data):
		valid_data = pos_valid_data + neg_valid_data
		best_threshold = 0
		best_f_score = 0
		best_accuracy = 0
		fn = 0.
                fp = 0.
                tp = 0.
                tn = 0.
                true = 0.
                false = 0.
		f = open('question','w')
		g = open('question_valid','w')
		z = open('question_test','w')
		count = 0
		for col in pos_valid_data:
			if count < (float(len(pos_valid_data)) * 0.7):
				for k in col: 
                        		for l in k.pos_words():
                                		f.write(l + ' ')
				f.write('\n')
			elif (count > (float(len(pos_valid_data)) * 0.7)) and (count < (float(len(pos_valid_data)) * 0.9)):
				for k in col:
                                        for l in k.pos_words():
                                                g.write(l + ' ')
                                g.write('\n')
			else:
                                for k in col:
                                        for l in k.pos_words():
                                                z.write(l + ' ')
                                z.write('\n')
			count += 1
			if self.baseline(col) == 1 or (self.perplexity(col) < 2.5):
				tp += 1
				true += 1
			else:
				fn += 1
				false += 1
		count = 0
		q = open('neg_test','w')
		for col in neg_valid_data:
			if count < (float(len(pos_valid_data)) * 0.1):
                                for k in col:
                                        for l in k.pos_words():
                                                q.write(l + ' ')
                                q.write('\n')
			count += 1
			if self.baseline(col) == 1 or (self.perplexity(col) < 2.5):
			#if self.perplexity(col) < 6:
				fp += 1
				false += 1
			else:
				tn += 1
				true += 1
		precision = tp/(tp+fp)
		recall = tp/(tp+fn)
		f_score = (2*precision*recall)/(precision+recall)
		if(true/(true+false) > best_accuracy):
			#best_threshold = threshold
			best_accuracy = true/(true+false)
			print 'accu' + str(best_accuracy)
			print 'precision' + str(precision)
			print 'recall' + str(recall)
			print 'f_score' + str(f_score) 
			#print 'threshold' + str(threshold)
		#return best_threshold
	def perplexity(self,sentence,type='utt'):
		if(type == 'utt'):
			input_ins,label_ins = self.generate_input(sentence)
		else:
			input_ins,label_ins = self.generate_sentence_input(sentence)
                if not label_ins:
                	return 1;
                input_ins = numpy.asarray(numpy.int32(input_ins))
                label_ins = numpy.asarray(numpy.int32(label_ins))
		softmax_array = self.rnn_model.prob(input_ins)
		prob = 1.0
		for i,j in zip(label_ins,softmax_array):
                	prob = prob*j[i]
		if prob == 0.0:
			return 100
		return (1/prob) ** (1/float(len(label_ins)))

test = swda_reader()
qa_model = pos_based_qa_model(test.transcript_reader())
qa_model.load_model(dic= 'dic')
qa_model.init_qa_models()
qa_model.load_model('pos_model','neg_model','dic')
qa_model.save_model(None,None,dic = 'dic')
qa_model.get_threshold()
'''
for i in range(10):
	qa_model.train_qa_models()
qa_model.save_model('pos_model','neg_model','dic')
'''
