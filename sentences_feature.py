import nltk
import string
import os
import re
from xmlparser_multiple import parse_bc3
import numpy
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from nltk.stem.porter import PorterStemmer
from evaluation import divide_data,weightRecall
from paragraph_vector_feature import readVectors,assignVectors
from clue_words import clue_score_calculation,repeat_words
from logistic import sgd_optimization_mnist
from python_rouge import rouge
from python_weka import python_weka
import weka.core.jvm as jvm
from rnn_summ_regression import rnn_summ
def stem_tokens(tokens, stemmer):
    	stemmed = []
    	for item in tokens:
        	stemmed.append(stemmer.stem(item))
    	return stemmed
def tokenize(text):
	stemmer = PorterStemmer()
    	tokens = nltk.word_tokenize(text)
    	stems = stem_tokens(tokens, stemmer)
    	return stems
def tfidf_generize(mails,token_dict):
	stemmer = PorterStemmer()
	tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
	mail_index = 0
	for i in mails:
		thread_index = 0
		for j in i.thread:
			name = 'mail_' + str(mail_index)+'thread_'+str(thread_index)+'_subject'
			lowers = i.subject[thread_index].lower()
			no_punctuation = lowers.translate(None, string.punctuation)
			token_dict[name] = no_punctuation
			sentence_index = 0
			for k in j.sentences:
				name = 'mail_' + str(mail_index)+'thread_'+str(thread_index)+'sentence_'+str(sentence_index)
				lowers = k.sentence.lower()
				no_punctuation = lowers.translate(None, string.punctuation)
				token_dict[name] = no_punctuation
			sentence_index += 1
			thread_index += 1
		mail_index += 1
	tfidf.fit_transform(token_dict.values())
	return tfidf
def calculate_sentence_features(mails):
	token_dict = {}
	stemmer = PorterStemmer()
	tfidf = tfidf_generize(mails,token_dict)
	for i in mails:
		for j in i.thread:
			for k in j.sentences:
				k.sentence_features = []
	max_sum = 0
	min_sum = 100
	max_avg = 0
	min_avg = 100
	max_sub_sim = 0
	min_sub_sim = 100
	max_clue_score = 0
	min_clue_score = 100
	#clue_score_calculation(mails,repeat_words)
	speech_act_score(mails)
	for i in mails:
		relative_position(i)
		tmp_max,tmp_min = subject_sim(i)
		if tmp_max > max_sub_sim:
			max_sub_sim = tmp_max
		if tmp_min < min_sub_sim:
			min_sub_sim = tmp_min
		message_number(i)
		t_index = 0
		for j in i.thread:
			tmp_subject = i.subject[t_index]
			subject_tfidf = numpy.array(tfidf.transform([tmp_subject]).todense())[0]
			t_index += 1
			m_rel_pos(j)
			fol_quote(j)
			for k in j.sentences:
				length(k)
				is_question(k)
				k.tfidf = numpy.array(tfidf.transform([k.sentence]).todense())[0]
				tfidf_sum(k)
				'''
				if k.tfidf_sum_score > max_sum:
					max_sum = k.tfidf_sum_score
				elif k.tfidf_sum_score <min_sum:
					min_sum = k.tfidf_sum_score
				'''
				tfidf_avg(k)
				#print tmp_subject
				k.tfidf_subject_similarity_score = cosine_similarity(k.tfidf,subject_tfidf)
				'''
				if k.tfidf_avg_score > max_avg:
					max_avg = k.tfidf_avg_score
				elif k.tfidf_avg_score < min_avg:
					min_avg = k.tfidf_avg_score
				if k.clue_score > max_clue_score:
					max_clue_score = k.clue_score
				elif k.clue_score < min_clue_score:
					min_clue_score = k.clue_score
				'''
	
	vectors = readVectors('bc3_vector_with_subject')
        assignVectors(mails,vectors)
        for i in mails:
                #print len(i.thread)
                #print len(i.thread_feature)
		subjectivity_score(i)
                t_index = 0
                for j in i.thread:
                        #print len(i.thread)
			question_similarity(j,paragraph_sim)	
                        j.vector = i.thread_feature[t_index]
                        j.subject_vector = i.subject_feature[t_index]
                        t_index += 1
			pre_sim(j,paragraph_sim)
                        #print t_index
                        for k in j.sentences:
                                for l in i.text:
                                        if k.index == l.index:
                                                k.vector = l.vector
                                                k.subject_similarity_score = cosine_similarity(k.vector,j.subject_vector)
                                                k.topic_similarity_score = cosine_similarity(k.vector,j.vector)
                                                break
	
	#input_x = []
	#input_y = []
	clue_score_calculation(mails,repeat_words,paragraph_sim)
	sentiment_score(mails,'bc3_sentiment_vectors.txt')
	for i in mails:
		for j in i.thread:
			for k in j.sentences:
				#k.subject_sim_score = (k.subject_sim_score - min_sub_sim)/float(max_sub_sim-min_sub_sim)
				#k.tfidf_sum_score = (k.tfidf_sum_score - min_sum) / float(max_sum - min_sum)
				#k.tfidf_avg_score = (k.tfidf_avg_score - min_avg) / float(max_avg - min_avg)
				k.unnor_sentence_features = [k.req,k.dlv,k.cmt,k.prop,k.meet,k.ddata,k.subjectivity_score,k.tfidf_subject_similarity_score,k.clue_score,k.subject_sim_score,k.message_number_score,k.fol_quote_score,k.m_rel_pos_score,k.is_question_score,k.tfidf_sum_score,k.tfidf_avg_score,k.length_score,k.relative_position_score]

				#k.clue_score = (k.clue_score - min_clue_score) / float(max_clue_score - min_clue_score)
				k.para_features = [k.subjectivity_score,k.qa_score,k.question_similarity_score,k.seq_sim_score,k.para_clue_score,k.fol_quote_score,k.is_question_score,k.tfidf_sum_score,k.tfidf_avg_score,k.subject_similarity_score,k.topic_similarity_score]
				#print k.subject_sim_score
				#k.sentence_features = [k.clue_score,k.subject_sim_score,k.message_number_score,k.fol_quote_score,k.m_rel_pos_score,k.is_question_score,k.tfidf_sum_score,k.tfidf_avg_score,k.length_score,k.relative_position_score]
				#print k.sentence_features
				#input_x.append(k.sentence_features)
				#input_y.append(k.score)
	unnor_label = {'subj':'real','tfidf_sim':'real','clue':'real','sub_sim':'real','m_num':'real','fol':'real','rel':'real','is_q':'real','tfidf_sum':'real','tfidf_avg':'real','leng':'real','rel_pos':'real','score':'real'}
	para_label = {'sen':'real','subj':'real','qa':'real','qs':'real','seq':'real','clue':'real','fol':'real','is_q':'real','tfidf_sum':'real','tfidf_avg':'real','subj_sim':'real','topic_sim':'real','score':'real'}
	
	'''
	clue_score_calculation(mails,paragraph_sim)
	for i in mails:
		for j in i.thread:
			for j in j.sentences:
				k.para_features = [k.clue_score,k.message_number_score,k.fol_quote_score,k.m_rel_pos_score,k.is_question_score,k.tfidf_sum_score,k.tfidf_avg_score,k.length_score,k.relative_position_score,k.subject_similarity_score,k.topic_similarity_score]
	'''
	#input_x = numpy.asarray(input_x)
	#input_y = numpy.asarray(input_y)
	# basic feature extraction end

	#clue_score
	
	#clue_score end
	'''
	vectors = readVectors('bc3_vector_with_subject')
	assignVectors(mails,vectors)
	for i in mails:
		#print len(i.thread)
		#print len(i.thread_feature)
		t_index = 0
		for j in i.thread:
			#print len(i.thread)
			j.vector = i.thread_feature[t_index]
			j.subject_vector = i.subject_feature[t_index]
			t_index += 1
			#print t_index
			for k in j.sentences:
				for l in i.text:
					if k.index == l.index:
						k.vector = l.vector
						k.subject_similarity_score = cosine_similarity(k.vector,j.subject_vector)
						k.topic_similarity_score = cosine_similarity(k.vector,j.vector)
						k.para_features = k.sentence_features
						k.para_features.append(k.subject_similarity_score)
						k.para_features.append(k.topic_similarity_score)
						break
	'''
	ori = 0.0
	para = 0.0
	o_ro = 0.0
	p_ro = 0.0
	rnn_rouge = 0.0
	rnn_score = 0.0
	for p in range(5):
		train,valid,test = divide_data(len(mails),0.8,0,mails)
		if p == 0:
			tmp_rnn = rnn_summ(train,test,"rnn_hidden")
		else:
			tmp_rnn = rnn_summ(train,test)
		tmp_rnn.init_rnn(0.01,0)
		rate = 0.33
		for j in range(6000):
			if j % 100 == 0:
				rate = rate * 0.9
			error = tmp_rnn.rnn_train(rate)
			if j % 100 == 0:
                        	print error
		if p==0:
			tmp_rnn.close_file()
		unnor_input_x = []
		unnor_test_x = []
		para_input_x = []
		para_test_x = []
		input_x = []
		input_y = []
		for i in train:
			for j in i.thread:
				for k in j.sentences:
					input_x.append(k.sentence_features)
					input_y.append(k.score)
					unnor_input_x.append(k.unnor_sentence_features)
					para_input_x.append(k.para_features)
		test_x = []
		test_y = []
		for i in test:
			for j in i.thread:
				for k in j.sentences:
					test_x.append(k.sentence_features)
					unnor_test_x.append(k.unnor_sentence_features)
					para_test_x.append(k.para_features)
					test_y.append(k.score)
		tmp_input = unnor_input_x + unnor_test_x
       	 	tmp_input = preprocessing.scale(tmp_input)
        	unnor_input_x = tmp_input[0:len(unnor_input_x)]
        	unnor_test_x = tmp_input[len(unnor_input_x):len(tmp_input)]

		tmp_input = para_input_x + para_test_x
		tmp_input = preprocessing.scale(tmp_input)
		para_input_x = tmp_input[0:len(para_input_x)]
		para_test_x = tmp_input[len(para_input_x):len(tmp_input)]
		def eval(input_x,input_y,test_x,test,label,write_folder = None):
			tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
              		  			'C': [1, 10, 100, 1000]},
                				{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
			grid_clf = GridSearchCV(SVR(C=1,epsilon=0.2), tuned_parameters)
			grid_clf.fit(input_x,input_y)
			print "params : \t"
			print grid_clf.get_params()
			result = grid_clf.predict(test_x)
			#py_weka = python_weka(input_x,input_y,label)
			#py_weka.train()
			#result = py_weka.predict(test_x)
			#py_weka.close()
			#clf = SVR(C=1.0, epsilon=0.2)
			#clf.fit(input_x,input_y)
			#result =  clf.predict(test_x)
			score_index = 0
			produce_set = []
			for i in test:
				produce_set.append([])
				score_list = []
				index_list = []
				for j in i.thread:
					for k in j.sentences:
						k.predict_score = result[score_index]
						score_index += 1
						score_list.append(k.predict_score)
						index_list.append(k.index)
				sorted_index_array = sorted_index(score_list)
				sen_length = 0
				for j in range(len(index_list)):
					if sen_length < float(len(index_list))*0.3:
						produce_set[-1].append(index_list[sorted_index_array[len(index_list)-j-1]])
						sen_length += 1
					else:
						break
			score =  weightRecall(test,produce_set,write_folder)
			print score
			rouge_eval = rouge(test,produce_set)
			rouge_score =  rouge_eval.eval()['rouge_l_f_score']
			print rouge_score
			return score,rouge_score
		print "rnn:"
		rnn_tmp_score,rnn_tmp_rouge = tmp_rnn.rnn_test()
		'''
		if p != 0:
			rnn_tmp_score,rnn_tmp_rouge = tmp_rnn.rnn_test()
		else:
			rnn_tmp_score,rnn_tmp_rouge = tmp_rnn.rnn_test("rnn_folder")
		'''
		rnn_score += rnn_tmp_score
		rnn_rouge += rnn_tmp_rouge
		print "avg rnn:"
		print rnn_score
		print "avg rnn rouge:"
		print rnn_rouge
		print "\n"
		print "ori:"
		if p == 0:
			ori_score,ori_rouge = eval(unnor_input_x,input_y,unnor_test_x,test,unnor_label,"ori_folder")
		else:
			ori_score,ori_rouge = eval(unnor_input_x,input_y,unnor_test_x,test,unnor_label)
		ori += ori_score
		o_ro += ori_rouge
		print "avg_ori:"
		print ori
		print "avg_ori_rouge:"
		print o_ro
		print "\n" 

		print "para"
		if p != 4:
			para_score,para_rouge = eval(para_input_x,input_y,para_test_x,test,para_label)
		else:
			para_score,para_rouge = eval(para_input_x,input_y,para_test_x,test,para_label,"para_folder")
		para += para_score
		p_ro += para_rouge
		print "avg_para:"
		print para
		print "avg_para_rouge:"
		print p_ro
		print "\n"
def speech_act_score(mails):
	for i in range(len(mails)):
		f = open('bc3_doc/mail_' + str(i) + '_speech')
		index = 0
		for line in f.readlines():
			count = 0
			sentence = None
			for j in mails[i].thread:
				if (len(j.sentences) + count) > index :
					for k in j.sentences:
						if count == index:
							sentence = k
							break
						else:
							count = count + 1
					break
				else:
					count = count + len(j.sentences)
			if line[0] == '*':
				index += 1
			else:
				if line.split()[2] == 'true':
					sign = 1
				elif line.split()[2] == 'false':
					sign = -1
					
				if line.split()[0] == 'Req':
					sentence.req = sign * float(line.split()[5])
				elif line.split()[0] == 'Dlv':
					sentence.dlv = sign * float(line.split()[5])
				elif line.split()[0] == 'Cmt':
					sentence.cmt = sign * float(line.split()[5])
				elif line.split()[0] == 'Prop':
					sentence.prop = sign * float(line.split()[5])
				elif line.split()[0] == 'Meet':
					sentence.meet = sign * float(line.split()[5])
				elif line.split()[0] == 'Ddata':
					sentence.ddata = sign * float(line.split()[5])
def sentiment_score(mails,index_file):
	g = open(index_file,'r')
	sentiment = []
	sentiment_index = []
	sentiment_prediction = sgd_optimization_mnist('train.txt','bc3_sentiment_vectors.txt')
	for i in sentiment_prediction:
		sentiment.append(i[1])
	for i in g.readlines():
		sentiment_index.append(i.split()[0])
	sentiment_dic = {}
	for i in range(len(sentiment)):
		sentiment_dic[sentiment_index[i]] = sentiment[i]
		print sentiment_index[i]
	count = 0
	ma = 0
	for i in mails:
		for j in i.thread:
			for k in j.sentences:
				index = '_mail' + str(ma) + '_' + str(k.index)
				#print index
				k.sentiment_score = sentiment_dic[index]
				count += 1	
		ma += 1
def subjectivity_score(mail):
	for i in mail.thread:
		for j in i.sentences:
			if j.index not in mail.subjectivity:
				j.subjectivity_score = 0
			else:
				j.subjectivity_score = mail.subjectivity[j.index]
def sorted_index(myList):
	return [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1])]
def paragraph_sim(s1,s2):
	return cosine_similarity(s1.vector,s2.vector)
def cosine_similarity(v1,v2):
	"compute cosine similarity of v1 to v2: (v1 dot v1)/{||v1||*||v2||)"
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(len(v1)):
		x = float(v1[i]); y = float(v2[i])
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	if sumxx == 0 or sumyy == 0:
		return 0
	return sumxy/math.sqrt(sumxx*sumyy)
def tfidf_sum(sentence):
	sentence.tfidf_sum_score =  sentence.tfidf.sum()
	#sentence.sentence_features.append(sentence.tfidf_sum_score)
def tfidf_avg(sentence):
	sentence.tfidf_avg_score = sentence.tfidf.mean()
	#sentence.sentence_features.append(sentence.tfidf_avg_score)
def length(sentence):
	sentence.length_score = len(sentence.sentence.split())
	#sentence.sentence_features.append(sentence)
def relative_position(mail):
	index = 0.0
	length = 0.0
	for i in mail.thread:
		length += len(i.sentences)
	for i in mail.thread:
		for j in i.sentences:
			j.relative_position_score = index/length
			index += 1
def is_question(sentence):
	gt = re.compile(r'\&gt\;')
	arrow = re.compile(r'\>')
	if len(sentence.sentence) > 0:
		gt_depth = len(gt.findall(sentence.sentence.split()[0]))
		arrow_depth = len(arrow.findall(sentence.sentence.split()[0]))
		if gt_depth >0 or arrow_depth>0:
			sentence.is_question_score = 0
			return 
	for i in sentence.sentence:
		if i == '?':
			sentence.is_question_score = 1
			return 
	sentence.is_question_score = 0
def question_similarity(thread,sim):
	for i in thread.sentences:
		i.question_similarity_score = 0
		if i.is_question_score == 1:
			i.question_similarity_score = 1
			continue
		for j in thread.sentences:
			if i != j and j.is_question_score == 1:
				i.question_similarity_score = max(i.question_similarity_score,sim(i,j))
def subject_sim(mail):
	def repeat_words(sen1,sen2):
		sum = 0
		for i in sen1:
			for j in sen2:
				if i == j:
					sum += 1
		return sum
	max_sim = 0
	min_sim = 100
	for i,j in zip(mail.thread,mail.subject): 
		for k in i.sentences:
			k.subject_sim_score = repeat_words(k.sentence,j)
			if k.subject_sim_score > max_sim:
				max_sim = k.subject_sim_score
			elif k.subject_sim_score < min_sim:
				min_sim = k.subject_sim_score
			#print k.subject_sim_score
	return (max_sim,min_sim)
'''
def num_of_res(mail):
	if 1:
		#do something

def num_of_recipents(message):
	if 1:
		#do something
'''
def fol_quote(thread):
	quote = 0
	for i in thread.sentences:
		if quote == 1:
			i.fol_quote_score = 1
		else:
			i.fol_quote_score = 0
		gt = re.compile(r'\&gt\;')
                arrow = re.compile(r'\>')
		sentence = i.sentence.split()
                if len(sentence)>0:
                	gt_depth = len(gt.findall(sentence[0]))
                       	arrow_depth = len(arrow.findall(sentence[0]))
			if gt_depth >0 or arrow_depth>0:
				quote = 1
		else:
			quote = 0
def message_number(mail):
	index = 0.0
	for i in mail.thread:
		for j in i.sentences:
			j.message_number_score = index/float(len(mail.thread))
		index += 1
def m_rel_pos(thread):
	index = 0.0
	for i in thread.sentences:
		i.m_rel_pos_score = index/float(len(thread.sentences))
		index += 1


def pre_sim(thread,sim_func):
	for i in range(len(thread.sentences)):
		if i == 0:
			if (i+1) < len(thread.sentences):
				thread.sentences[i].seq_sim_score = sim_func(thread.sentences[i],thread.sentences[i+1])
			else:
				thread.sentences[i].seq_sim_score = 0.0
		elif i == (len(thread.sentences) - 1):
			thread.sentences[i].seq_sim_score = sim_func(thread.sentences[i],thread.sentences[i-1])
		else:
			thread.sentences[i].seq_sim_score = max(sim_func(thread.sentences[i],thread.sentences[i+1]),sim_func(thread.sentences[i],thread.sentences[i-1]))
'''
def next_sim(thread,sim_func):

def centroid_sim(thread,sim_func):
'''
corpus = 'bc3/bc3corpus.1.0/corpus.xml'
annotation = 'bc3/bc3corpus.1.0/annotation.xml'
mails = parse_bc3(corpus,annotation)
#jvm.start()
calculate_sentence_features(mails)
#jvm.stop()
