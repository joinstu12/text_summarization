from nltk.corpus import stopwords
import numpy
import string
def phrase_extraction(vec):
	stoplist = stopwords.words('english')
	phrase_list = [[]]
	vec_phrase_index = []
	for i in vec:
		tmp_phrase_index = []
		if(len(phrase_list[-1])>0):
			phrase_list.append([])
		#sentence_delimiters = re.compile(u'[.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
		tmp = i.sentence.lower().strip().split()
		#tmp = re.split('( +)|\.',i.sentence.lower())
		#tmp = sentence_delimiters.split(tmp)
		for j in range(len(tmp)):
			if((tmp[j] in string.punctuation) or (tmp[j] in stoplist)):
				tmp[j] = '|'
		for j in tmp:
			if(j == '|' and len(phrase_list[-1])>0):
				tmp_phrase_index.append(len(phrase_list)-1)
				phrase_list.append([])
			elif(j == '|'):
				continue
			else:
				tmp_word = ''
				punct = 0
				for k in range(len(j)):
					if(j[k] not in string.punctuation):
						tmp_word += j[k]
					elif(k == len(j)-1 and (j[k]==',' or j[k] == '.' or j[k] == '?' or j[k] == '!')):
						punct = 1	
				phrase_list[-1].append(tmp_word)
				if(punct == 1):
					tmp_phrase_index.append(len(phrase_list)-1)
					phrase_list.append([])
		vec_phrase_index.append(tmp_phrase_index)
	return phrase_list,vec_phrase_index
class word_struct:
	def __init__(self,deg=0,freq=0):
		self.deg = deg
		self.freq = freq
		self.score = 0
def word_score_calculate(phrase_list):
	word_list = {}
	for i in phrase_list:
		for j in i:
			if(j not in word_list):
				word_list[j] = word_struct(len(i)-1+1,1)
			else:
				word_list[j].deg += len(i)-1+1
				word_list[j].freq += 1
	for i in word_list:
		word_list[i].score = word_list[i].deg/word_list[i].freq
	return word_list
