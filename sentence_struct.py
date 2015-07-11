import re
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
class sentence_struct:
        def __init__(self,sentence,index):
                self.sentence = sentence
                self.index = index
                self.remove_stop_ver = stop_word_remove(sentence)
		self.score = 0
		self.vector = None
		self.prediction = 0
		self.clue_score = 0
def stop_word_remove(sentence):
	URL_REGEX = re.compile(r'''((?:mailto:|ftp://|http://)[^ <>'"{}|\\^`[\]]*)''')
	sentence = URL_REGEX.sub(' ',sentence)
        stemmer = PorterStemmer()
        #stoplist = set('for a of the and to in'.split())
        stoplist = stopwords.words('english')

        ret_sen = []
        #sentence = str(sentence).translate(None, string.punctuation)
        #sentence = tokenize(sentence.lower())
        #table = string.maketrans("","")
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        sentence = regex.sub('', sentence)
        sentence = sentence.lower().strip().split()
        for i in range(len(sentence)):
                token = ''
                for j in sentence[i]:
                        if(j != '.' ):
                                token += j
                sentence[i] = token
                #sentence[i] = stemmer.stem(token)
        for i in range(len(sentence)):
                if(sentence[i] not in stoplist):
                        ret_sen.append(sentence[i])
        return ret_sen

