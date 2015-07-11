import xml.etree.ElementTree as ET
import os
import re
from tfidf import tfidf
from sentence_struct import sentence_struct
from sentence_struct import stop_word_remove
class thread_struct:
	def __init__(self,number):
		self.number = number
		self.sentences = []
		self.fragment = []
class mail_struct:
	def __init__(self,number):
		self.name = None
		self.text = []
		self.vector = None
		self.annotations = []
		self.subjectivity = {}
		self.subject = []
		self.thread = []
		self.thread_feature = []
		self.remove_stop_ver_title = None
		self.subject_feature = []
		self.annotators = []
		self.number = number
def parse_file(file):
        tree = ET.parse(file)
        root = tree.getroot()
	mails = []
	space = re.compile(r' ',re.S)
	sp = re.compile(r'\-+',re.S)
	gt = re.compile(r'\&gt',re.S)
        for i in root:
		tmp = mail_struct(len(mails))
		for j in i:
			if(j.tag == 'name'):
				#print(space.sub('_',j.text))
				tmp.name = space.sub('_',j.text)
				tmp.remove_stop_ver_title = stop_word_remove(j.text)
				#print(j.text)
			elif(j.tag == 'DOC'):
				thread_num = -1
				for k in j:
					if k.tag == 'Subject':
						tmp.subject.append(k.text)
					if(k.tag == 'Text'):
						for l in k:
							res = gt.search(l.text)
							'''
							if(res is not None or l.text[0] == '>' or(l.text[0] =='_' and l.text[1] == '__')):
								continue
							if(l.text[0] == 'h' and l.text[1]=='t' and l.text[2] == 't' and l.text[3] == 'p'):
								continue
							'''
							tmp_text = l.text.replace('\n',' ')
							tmp_text = sp.sub('',tmp_text)
							if(tmp_text.isspace()):
								continue
							t_s = sentence_struct(tmp_text,l.attrib['id'])
							tmp.text.append(t_s)

							if t_s.index.split('.')[0] != thread_num:
								thread_num = t_s.index.split('.')[0]
								tmp.thread.append(thread_struct(int(thread_num)))
								#thread_num = t_s.index.split('.')[0]
							tmp.thread[-1].sentences.append(t_s)
							#print t_s.sentence
							#tmp.id.append(l.attrib['id'])
							#print(l.text)
		mails.append(tmp)
	for i in mails:
		for j in i.text:
			for k in range(len(j.remove_stop_ver)):
				if(len(j.remove_stop_ver[k])<2 or j.remove_stop_ver[k] == 'gt' or j.remove_stop_ver[k] == 'gtgt'):
					j.remove_stop_ver[k] = ''
			#print(j.remove_stop_ver)
			j.remove_stop_ver = (filter(None,j.remove_stop_ver))
	return mails
def parse_anno(file,mails):
	tree = ET.parse(file)
	root = tree.getroot()
	space = re.compile(r' ',re.S)
	for i in root:
		for j in i:
			#print(j.tag)
			if(j.tag == 'name'):
				for k in mails:
					tmp_name = space.sub('_',j.text)
					if(k.name == tmp_name):
						tmp = k
						break
			elif(j.tag == 'annotation'):
				for k in j:
					if(k.tag == 'desc'):
						tmp.annotators.append([])
					if(k.tag == 'sentences'):
						for l in k:
							if(l.tag == 'sent'):
								tmp.annotators[-1].append(l.attrib['id'])
								tmp.annotations.append(l.attrib['id'])
							if(l.tag == 'subj'):
								if l.attrib['id'] not in tmp.subjectivity:
									tmp.subjectivity[l.attrib['id']] = 1
								else:
									tmp.subjectivity[l.attrib['id']] += 1
	return mails
def calculate_score(mails):
	max = 0
	for i in mails:
		for j in i.text:
			for k in i.annotations:
				if(j.index == k):
					j.score += 1.0
					if(j.score > max):
						max = j.score
	for i in mails:
		for j in i.text:
			j.score = j.score/max
def parse_bc3(corpus,annotation):
	mails = parse_file(corpus)
	mails = parse_anno(annotation,mails)
	calculate_score(mails)
	return mails

'''
mails = parse_file('bc3/bc3corpus.1.0/corpus.xml')
mails = parse_anno('bc3/bc3corpus.1.0/annotation.xml',mails)
test = tfidf()
for i in mails:
	for j in i.text:
		test.addDocument(j.index,j.remove_stop_ver)
for i in mails:
	for j in i.text:
		print test.similarities(j.remove_stop_ver)
'''
corpus = 'bc3/bc3corpus.1.0/corpus.xml'
annotation = 'bc3/bc3corpus.1.0/annotation.xml'
parse_bc3(corpus,annotation)
'''
mails = parse_file(corpus)
mails = parse_anno(annotation,mails)
calculate_score(mails)
'''
'''
for i in mails:
	for j in i.text:
		print(j.sentence)
'''
'''
for i in mails:
	print i.annotations
	for j in i.text:
		print(j.index)
'''

