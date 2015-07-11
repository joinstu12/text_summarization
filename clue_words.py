from xmlparser_multiple import parse_bc3
import re
import numpy
from nltk.stem.porter import PorterStemmer

class fragment_struct():
	def __init__(self,depth):
		self.depth = depth
		self.sentences = []
		self.number = None
		self.clue_word_list = {}
class clue_struct():
	def __init__(self,mail):
		self.fragment = []
		self.data = mail
		self.graph = None
		self.sorted_fragment = []
	def fragmentation(self):
		def quotation_depth(sentence):
                	depth = 0
			gt = re.compile(r'\&gt\;')
			arrow = re.compile(r'\>')
      			if len(sentence)>0:
                    		gt_depth = len(gt.findall(sentence[0]))
				arrow_depth = len(arrow.findall(sentence[0]))
				if gt_depth>0:
					depth = gt_depth
				if arrow_depth>0:
					depth = arrow_depth
                	return depth
		for i in self.data.thread:
			now_depth = None
			for j in i.sentences:
				if now_depth != quotation_depth(j.sentence.split()):
					now_depth = quotation_depth(j.sentence.split())
					i.fragment.append(fragment_struct(now_depth))
				i.fragment[-1].sentences.append(j)
		clear = 0
		def split_equal_fragment_in_thread(thread1,thread2):
			clear = 1
			for i in range(len(thread1.fragment)):
				for j in range(len(thread2.fragment)):
					#print(thread1.fragment[i])
					if (thread1.fragment[i].depth == 0 and thread2.fragment[j].depth>0) or (thread1.fragment[i].depth > 0 and thread2.fragment[j].depth == 0):
						equal_list = self.detect_equal_subfragment(thread1.fragment[i],thread2.fragment[j])
						if equal_list:
							clear = 0
							self.split_equal_subfragment(thread1,thread2,i,j,equal_list)
			return clear	
		while(clear == 0):
			clear = 1
			for i in range(len(self.data.thread)):
				#print self.data.thread[i].fragment[0]
				for j in range(len(self.data.thread) - i -1):
					if split_equal_fragment_in_thread(self.data.thread[i],self.data.thread[i+j+1]) == 0:
						clear = 0
					
					'''
					equal_list = self.detect_equal_subfragment(self.data.thread[i],self.data.thread[i+j+1])
					if len(equal_list) > 0 and fragment_is_equal(self.data.thread[i],self.data.thread[i+j+1]):
						clear = 0
						split_euqal_subfragment(self.data.thread[i],self.data.thread[i+j+1],equal_list)
					'''		
		'''
		for i in range(len(self.data.thread)):
			for j in range(len(self.data.thread)):
				if i != j:
					#print('thread'+str(i) + ' and thread' + str(j))
					for k in self.data.thread[i].fragment:
						for l in self.data.thread[j].fragment:
							self.split_fragment(k,l)
		'''
	def split_equal_subfragment(self,t1,t2,f1,f2,equal_list):
		def insert_fragment_in_index(fragment,fragment_insert,index,depth):
			if(len(fragment_insert)>0):
				new_fragment = fragment_struct(depth)
				new_fragment.sentences = fragment_insert
				fragment.insert(index,new_fragment)
		for i in equal_list:
			length = i[1][0] - i[0][0] + 1
			new_fragment_1_1 = t1.fragment[f1].sentences[0:i[0][0]]
			new_fragment_1_2 = t1.fragment[f1].sentences[i[0][0] : (i[1][0]+1)]
			new_fragment_1_3 = t1.fragment[f1].sentences[(i[1][0] + 1):len(t1.fragment[f1].sentences) ]
			new_fragment_2_1 = t2.fragment[f2].sentences[0:i[0][1]]
			new_fragment_2_2 = t2.fragment[f2].sentences[i[0][1]:(i[1][1]+1)]
			new_fragment_2_3 = t2.fragment[f2].sentences[(i[1][1]+1):len(t2.fragment[f2].sentences)]
			depth_1 = t1.fragment[f1].depth
			depth_2 = t2.fragment[f2].depth
			del(t1.fragment[f1])
			del(t2.fragment[f2])
			thread1_list = [new_fragment_1_1,new_fragment_1_2,new_fragment_1_3]
			thread2_list = [new_fragment_2_1,new_fragment_2_2,new_fragment_2_3]
			for i in thread1_list:
				insert_fragment_in_index(t1.fragment,i,f1,depth_1)
			for i in thread2_list:
                        	insert_fragment_in_index(t2.fragment,i,f2,depth_2)
			
	def fragment_is_equal(self,f1,f2):
		def sentence_is_equal(s1,s2):
			length = min(len(s1.remove_stop_ver),len(s2.remove_stop_ver))
			if length == 0: 
				if len(s1.remove_stop_ver) == len(s2.remove_stop_ver) :
					return True
				else:
					return False
			same = 0.0
			for i in range(length):
				if(s1.remove_stop_ver[i] == s2.remove_stop_ver[i]):
					same += 1
			if(same/float(length) >0.95):
				return True
			else:
				return False
		if len(f1.sentences) != len(f2.sentences):
			return False
		for i in range(len(f1.sentences)):
			if sentence_is_equal(f1.sentences[i],f2.sentences[i]) is False:
				return False
		return True
	def detect_equal_subfragment(self,fragment1,fragment2):
		if self.fragment_is_equal(fragment1,fragment2) is True:
			return []
		equal_head = (-1,-1)
		equal_list = []
		def in_equal_list(index_i,index_j):
			for i in equal_list:
				if index_i>=i[0][0] and index_i<= i[1][0] and index_j >= i[0][1] and index_j <= i[1][1]:
					return True
			return False
		for i in range(len(fragment1.sentences)):
			for j in range(len(fragment2.sentences)):
				if in_equal_list(i,j):
					continue
				if sentence_is_equal(fragment1.sentences[i],fragment2.sentences[j]):
					equal_head = (i,j)
					for k in range(min((len(fragment1.sentences)-i),(len(fragment2.sentences)-j))):
						if sentence_is_equal(fragment1.sentences[i+k],fragment2.sentences[j+k]):
							if k == (len(fragment1.sentences)-i-1) or k == (len(fragment2.sentences)-j-1):
								equal_end = (i+k,j+k)
								equal_list.append([equal_head,equal_end])
						else:
							equal_end = (i+k-1,j+k-1)
							equal_list.append([equal_head,equal_end])
							break
		#print equal_list
		return equal_list
	def assign_fragment_number(self):
		stack = []
		for i in self.data.thread:
			for j in i.fragment:
				if len(stack) == 0:
					stack.append(j)
					j.number = 0
				else:
					for k in stack:
						if self.fragment_is_equal(j,k) is True:
							j.number = k.number
							break
					if(j.number is None):
						stack.append(j)
						j.number = len(stack) - 1
		self.sorted_fragment = stack
		self.graph = numpy.zeros((len(stack),len(stack)))
		'''
		for i in self.data.thread:
			print 'thread' + str(i.number)
			for j in i.fragment:
				print j.number
				for k in j.sentences:
					print k.sentence
			print '\n'
		'''
	def construct_fragment_graph(self):
		for i in self.data.thread:
			for j in range(len(i.fragment)):
				for k in range(len(i.fragment)-j-1):
					if(i.fragment[j].depth == (i.fragment[k+j+1].depth - 1)):
						self.graph[i.fragment[j].number][i.fragment[k+j+1].number] = 1
					if(i.fragment[k+j+1].depth == (i.fragment[j].depth - 1)):
						self.graph[i.fragment[k+j+1].number][i.fragment[j].number] = 1
	def construct_sentence_graph(self,sim):
		index_list = []
		for i in self.data.thread:
			for j in i.sentences:
				index_list.append(j.index)
		self.sentence_graph = numpy.zeros((len(index_list),len(index_list)))
		for i in range(len(self.sorted_fragment)):
			for j in range(len(self.sorted_fragment)):
				if self.graph[i][j] == 1 and i != j:
					for k in self.sorted_fragment[i].sentences:
						for l in self.sorted_fragment[j].sentences:
							self.sentence_graph[index_list.index(k.index)][index_list.index(l.index)] = sim(k,l)
	def general_clue_score(self):
		index_list = []
		for i in self.data.thread:
			for j in i.sentences:
				index_list.append(j.index)
		for i in self.data.thread:
			for j in i.sentences:
				j.clue_score = 0.0
				for k in range(len(index_list)):
					if self.sentence_graph[index_list.index(j.index)][k] >0:
						j.clue_score += self.sentence_graph[index_list.index(j.index)][k]
					if self.sentence_graph[k][index_list.index(j.index)] >0:
						j.clue_score += self.sentence_graph[k][index_list.index(j.index)]
				#print j.clue_score
	def para_clue_score(self):
		def is_question(sentence):
			sentence = sentence.sentence
			for i in sentence:
				if i == '?':
					return True
			return False
                index_list = []
		sentence_list = []
                for i in self.data.thread:
                        for j in i.sentences:
                                index_list.append(j.index)
				sentence_list.append(j)
                for i in self.data.thread:
                        for j in i.sentences:
                                j.para_clue_score = 0.0
				j.qa_score = 0.0
                                for k in range(len(index_list)):
                                        if self.sentence_graph[index_list.index(j.index)][k] >0:
                                                j.para_clue_score += self.sentence_graph[index_list.index(j.index)][k]
						if is_question(sentence_list[k]):
							j.qa_score = max(j.qa_score,self.sentence_graph[index_list.index(j.index)][k])
                                        if self.sentence_graph[k][index_list.index(j.index)] >0:
                                                j.para_clue_score += self.sentence_graph[k][index_list.index(j.index)]
					'''
					#Question Answering part
					if self.sentence_graph[index_list.index(j.index)][K]>0 and is_question(sentence_list[k]):
                                                j.qa_score = max(j.qa_score,self.sentence_graph[index_list.index(j.index)][k]
					
					if self.sentence_graph[index_list.index(j.index)][K]>0 and is_question(sentence_list[k]):
						j.question_answer_score = max(j.question_answer_score,self.sentence_graph[index_list.index(j.index)][k]
					'''
	def detect_clue_words(self):
		def construct_clue_list(f1):
			for i in f1.sentences:
				for j in i.remove_stop_ver:
					if j not in f1.clue_word_list:
						f1.clue_word_list[j] = 0
		def calculate_clue_score(f1,f2):
			for i in f2.sentences:
				for j in i.remove_stop_ver:
					if j in f1.clue_word_list:
						f1.clue_word_list[j] += 1
		def calculate_sentence_score(sentence,clue_word_list):
			score = 0
			for i in sentence.split():
				if i in clue_word_list:
					score += clue_word_list[i]
			return score	
		for i in range(len(self.sorted_fragment)):
			construct_clue_list(self.sorted_fragment[i])
			for j in range(len(self.graph[i])):
				if self.graph[i][j] == 1: #parent node
					calculate_clue_score(self.sorted_fragment[i],self.sorted_fragment[j])
				if self.graph[j][i] == 1: #child noe
					calculate_clue_score(self.sorted_fragment[i],self.sorted_fragment[j])
			for j in self.sorted_fragment[i].sentences:
				for k in self.data.thread:
					for l in k.sentences:
						if l.index == j.index:
							l.clue_score = calculate_sentence_score(l.sentence,self.sorted_fragment[i].clue_word_list)
							#print l.clue_score
							break	
def repeat_words(s1,s2):
	stemmer = PorterStemmer()
	def stem_token(tokens, stemmer):
    		stemmed = []
    		for item in tokens:
        		stemmed.append(stemmer.stem(item))
    		return stemmed
	stem_1 = stem_token(s1.remove_stop_ver,stemmer)
	stem_2 = stem_token(s2.remove_stop_ver,stemmer)
	sim = 0.0
	for i in stem_1:
		for j in stem_2:
			if i == j:
				sim += 1
	return sim
def sentence_is_equal(s1,s2):
	length = min(len(s1.remove_stop_ver),len(s2.remove_stop_ver))
	
       	if length == 0:
        	if len(s1.remove_stop_ver) == len(s2.remove_stop_ver) :
                	return True
                else:
                      	return False
	if (len(s1.remove_stop_ver) / len(s2.remove_stop_ver)) < 0.9 or (len(s2.remove_stop_ver) / len(s1.remove_stop_ver)) < 0.9:
                return False
	same = 0.0
        for i in range(length):
        	if(s1.remove_stop_ver[i] == s2.remove_stop_ver[i]):
                	same += 1
        if(same/float(length) >0.95):
		#print s1.sentence + s2.sentence
        	return True
        else:
              	return False
def clue_score_calculation(mails,sim,para_sim):
	for i in mails:
		tmp = clue_struct(i)
		tmp.fragmentation()
		tmp.assign_fragment_number()
		tmp.construct_fragment_graph()
		tmp.construct_sentence_graph(sim)
		tmp.general_clue_score()
		for j in range(len(i.thread)):
			#print i.thread[j].fragment
                	for k in range(len(i.thread[j].sentences)):
                        	i.thread[j].sentences[k].clue_score = tmp.data.thread[j].sentences[k].clue_score
				#print i.thread[j].sentences[k].clue_score
		tmp.construct_sentence_graph(para_sim)
		tmp.para_clue_score()
		for j in range(len(i.thread)):
                        #print i.thread[j].fragment
                        for k in range(len(i.thread[j].sentences)):
                                i.thread[j].sentences[k].para_clue_score = tmp.data.thread[j].sentences[k].para_clue_score
		

'''	
corpus = 'bc3/bc3corpus.1.0/corpus.xml'
annotation = 'bc3/bc3corpus.1.0/annotation.xml'
mails = parse_bc3(corpus,annotation)

clue = clue_struct(mails[0])
count = 0
clue_score_calculation(mails,repeat_words)
clue_score_calculation(mails,repeat_words)
'''
'''
for i in mails:
	print count
	count += 1
	tmp = clue_struct(i)
	tmp.fragmentation()
	tmp.assign_fragment_number()
	tmp.construct_fragment_graph()
	tmp.construct_sentence_graph(repeat_words)
	tmp.general_clue_score()
	for j in range(len(i.thread)):
		for k in range(len(i.thread[j].sentences)):
			i.thread[j].sentences[k].clue_score = tmp.data.thread[j].sentences[k].clue_score
'''
	#tmp.detect_clue_words()
#clue.fragmentation()	
#for i in range(len(clue.data.thread)):
	
#clue.assign_fragment_number()
