from crawl_ted import crawl_article
from lexrank import lexrank
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from numpy.linalg import norm
from subprocess import call
import os
import gensim
import numpy
import string
import matplotlib.pyplot as plt
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
from rouge_eval import summary_eval
import shutil
from tfidf import tfidf
from sentence_struct import sentence_struct,stop_word_remove
from xmlparser import parse_file
from xmlparser import parse_anno
from xmlparser import mail_struct
from lda_process import lda_process
import math
from phrase_extraction import phrase_extraction,word_score_calculate
def construct_sentence_vec(article):
	index = 0
	sentence_vec = []
	for i in article:
		for j in i:
			sentence_vec.append(sentence_struct(j,index))
			index += 1
	return sentence_vec
def construct_2sen_vec(article):
	index = 0
        sentence_vec = []
        for i in article:
                for j in range(len(i)-1):
                        sentence_vec.append(sentence_struct(i[j]+i[j+1],index))
                        index += 1
        return sentence_vec
def construct_similarity_matrix(sim_file,vec_len):
	matrix = numpy.zeros((vec_len,vec_len))
	title_sim = numpy.zeros(vec_len)
	f = open(sim_file,'r')
	for i in range(vec_len):
		for j in range(vec_len):
			matrix[i][j] = float(f.readline())
	for i in range(vec_len):
		title_sim[i] = float(f.readline())
	return matrix,title_sim
def tokenize(documents):
	return PunktWordTokenizer().tokenize(documents)


def process_document(title,sen2=1,exist=1):
	article,art_title = crawl_article(title)
        #sen2 = 1
        #exist = 1
        if(sen2 == 1):
                input_name = title+'_sen2_test.txt'
                feature_name = title+'_sen2_features.txt'
                output_name = title+'_sen2_output.txt'
                vec = construct_2sen_vec(article)
	else:
                input_name = title+'_test.txt'
                feature_name = title+'_features.txt'
                output_name = title+'_output.txt'
                vec = construct_sentence_vec(article)
	'''
	if(exist == 0):
                construct_input_file(vec,input_name,art_title)
                os.system('python takelab_sts/takelab_simple_features.py '+input_name+' > '+feature_name)
                os.system('svm-predict '+feature_name+' model.txt ' +  output_name)
                os.system('python postprocess_scores.py '+ input_name+' ' + output_name)
        matrix,title_sim = construct_similarity_matrix(output_name,len(vec))
	'''
	tmp = mail_struct()
	tmp.name = title
	tmp.text = vec
	bot = re.compile(r'\_',re.S)
	tmp.remove_stop_ver_title = stop_word_remove(bot.sub(' ',title))
	matrix,title_sim = cal_similarity(tmp)
        important =  lexrank(matrix,len(vec),0.001)
	for k in range(len(important)):
		if(math.isnan(important[k])):
			important[k] = 0
        important = (important-min(important))/(max(important)-min(important))
	title_sim = (title_sim-min(title_sim))/(max(title_sim)-min(title_sim))
        sort_index = numpy.argsort(important)[::-1]
	return vec,important,title_sim
def cal_similarity(mail):
	tmp_tfidf = tfidf()
	similarity = []
	title_sim = []
	for i in mail.text:
		tmp_tfidf.addDocument(i.index,i.remove_stop_ver)
	for i in mail.text:
		tmp_vector = []
		tmp = tmp_tfidf.similarities(i.remove_stop_ver)
		for j in tmp:
			tmp_vector.append(j[1])		
		similarity.append(tmp_vector)
	#print(similarity)
	tmp_ti = tmp_tfidf.similarities(mail.remove_stop_ver_title)
	for i in tmp_ti:
		title_sim.append(i[1])
	return numpy.asarray(similarity),numpy.asarray(title_sim)
def cue_word(vec):
	cue_word_list = ['summary','in conclusion','propose','argue']
	cue_word_score = numpy.zeros(len(vec))
	for i in range(len(vec)):
		for j in cue_word_list:
			if(j in vec[i].remove_stop_ver):
				cue_word_score[i] = 1
	return cue_word_score
def process_mail(mail,exist=0):
	title = mail.name
	art_title = title
	vec = mail.text
	sen2 = 0
	if(sen2 == 1):
                input_name = title+'_sen2_test.txt'
                feature_name = title+'_sen2_features.txt'
                output_name = title+'_sen2_output.txt'
                #vec = construct_2sen_vec(article)
        else:
                input_name = title+'_test.txt'
                feature_name = title+'_features.txt'
                output_name = title+'_output.txt'
                #vec = construct_sentence_vec(article)
	matrix,title_sim = cal_similarity(mail)
        important =  lexrank(matrix,len(vec),0.001)
        important = (important-min(important))/(max(important)-min(important))
	if(max(title_sim)-min(title_sim) != 0):
        	title_sim = (title_sim-min(title_sim))/(max(title_sim)-min(title_sim))
	if(math.isnan(important[0])):
		print('similarity')
		print(matrix)
	for k in range(len(important)):
                if(math.isnan(important[k])):
                        important[k] = 0
        return vec,important,title_sim
def anno(mail):
	anno_summary = []
	haved = []
	for i in mail.annotations:
		for j in haved:
			if(j==i):
				continue
		haved.append(i)
		for j in mail.text:
			if(j.index == i):
				anno_summary.append(j.sentence)
				#print(j.sentence)
				break
	return anno_summary
def important_word(vec,word_list,phrase_list,vec_phrase_index):
	phrase_score = numpy.zeros(len(phrase_list))
	for i in range(len(phrase_list)):
		for j in phrase_list[i]:
			phrase_score[i] += word_list[j].score
	#print(phrase_list)
	#print(phrase_score)
	sort_list = numpy.argsort(phrase_score)[::-1]
	set_len = 15
	vec_phrase_score = numpy.zeros(len(vec))
	for i in range(len(vec_phrase_index)):
		for j in vec_phrase_index[i]:
			if(j in sort_list[0:set_len]):
				vec_phrase_score[i] += 1
	vec_phrase_score = vec_phrase_score / set_len
	#print(phrase_list[sort_list[0]])
	return vec_phrase_score
def bc3_eval():
	corpus = 'bc3/bc3corpus.1.0/corpus.xml'	
	annotation = 'bc3/bc3corpus.1.0/annotation.xml'
	mails = parse_file(corpus)
	mails = parse_anno(annotation,mails)
	sample_vector = []
        target_vector = []
        precision_vector = []
        recall_vector = []
        F_measure_vector = []
	imp = 0
        ti_s = 0
        to_s = 0
	train = 20
	index =0
	for i in mails:
		if(index>=train):
			break
		index += 1
		vec,important,title_sim = process_mail(i)
		topic_similarity = lda_process(vec)
		phrase_list,vec_phrase_index = phrase_extraction(vec)
		word_list = word_score_calculate(phrase_list)
		vec_phrase_score = important_word(vec,word_list,phrase_list,vec_phrase_index)
		tmp_produce = []
		standard_summary = [anno(i)]
		standard_name = [[]]
		cue_word_score = cue_word(vec)
		for j in range(len(standard_summary[0])):
			standard_name[0].append('PythonROUGE/'+i.name+'/'+str(j)+'_standard.txt')
		#print(standard_name)
		#standard_name[0].append('PythonROUGE/'+i.name+'/'+str(0)+'_standard.txt')
		#print(standard_name)
                #standard_name = [['PythonROUGE/'+i+'_standard.txt']]
                newpath = 'PythonROUGE/'+i.name
                if not os.path.exists(newpath):
                        os.makedirs(newpath)
                for j in range(len(vec)):
                        produce_name = ['PythonROUGE/'+i.name+'/'+str(j)+'.txt']
                        produce_summary = [[vec[j].sentence]]
                        sample_vector.append([important[j],title_sim[j],topic_similarity[j],vec_phrase_score[j],cue_word_score[j]])
                        imp += important[j]
                        ti_s += title_sim[j]
                        to_s += topic_similarity[j]
                        recall,precision,F_measure = summary_eval(standard_summary,standard_name,produce_summary,produce_name)
                        target_vector.append(recall[0]*precision[0]*F_measure[0])
                        recall_vector.append(recall[0])
                        precision_vector.append(precision[0])
                        F_measure_vector.append(F_measure[0])
		shutil.rmtree(newpath)
	for i in sample_vector:
                i[0] = i[0]/imp
                i[1] = i[1]/ti_s
                i[2] = i[2]/to_s
        x_recall_train, x_recall_test, y_recall_train, y_recall_test = cross_validation.train_test_split(sample_vector,recall_vector, test_size=0.2, random_state=0)
        x_precision_train, x_precision_test, y_precision_train, y_precision_test = cross_validation.train_test_split(sample_vector,precision_vector, test_size=0.2, random_state=0)
        x_fmeasure_train, x_fmeasure_test, y_fmeasure_train, y_fmeasure_test = cross_validation.train_test_split(sample_vector,F_measure_vector, test_size=0.2, random_state=0)
        #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3, 1e-4,1e-5],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        r_clf =  GridSearchCV(SVR(C=1,epsilon=0.2), tuned_parameters, cv=5)
        p_clf =  GridSearchCV(SVR(C=1,epsilon=0.2), tuned_parameters, cv=5)
        f_clf =  GridSearchCV(SVR(C=1,epsilon=0.2), tuned_parameters, cv=5)
	r_clf.fit(sample_vector,recall_vector)
        p_clf.fit(sample_vector,precision_vector)
        f_clf.fit(sample_vector,F_measure_vector)
	index = 0
	produce_summary = []
	produce_name = []
	standard_summary = []
	standard_name = []
	lex_summary = []
	lex_name = []
	for i in mails:
		if(index<train):
			index += 1
			continue
		if(i.name == 'Re:_StarOffice' or i.name == 'Try_Unsubscribing&ndash;&ndash;You_Can\'t'):
			continue
		#print(i.name)
		vec,important,title_sim = process_mail(i)
                topic_similarity = lda_process(vec)
		phrase_list,vec_phrase_index = phrase_extraction(vec)
                word_list = word_score_calculate(phrase_list)
                vec_phrase_score = important_word(vec,word_list,phrase_list,vec_phrase_index)
		cue_word_score = cue_word(vec)
                #word_list = word_score_calculate(phrase_extraction(vec))
                #print(word_list)
                tmp_produce = []
                standard_summary.append(anno(i))
                tmp_name = []
                for j in range(len(standard_summary[-1])):
                        tmp_name.append('PythonROUGE/'+i.name+'/'+str(j)+'_standard.txt')
		standard_name.append(tmp_name)
		newpath = 'PythonROUGE/'+i.name
                if not os.path.exists(newpath):
                        os.makedirs(newpath)
		maxs = 0
        	maxi = 0
		tmp_summary = []
		predict_rouge = []
        	for j in range(len(vec)):
                        #tmp = r_clf.predict([important[j],title_sim[j],topic_similarity[j],vec_phrase_score[j]])*p_clf.predict([important[j],title_sim[j],topic_similarity[j],vec_phrase_score[j]])*f_clf.predict([important[j],title_sim[j],topic_similarity[j],vec_phrase_score[j]])
                        tmp = f_clf.predict([important[j],title_sim[j],topic_similarity[j],vec_phrase_score[j],cue_word_score[j]])
			predict_rouge.append(tmp)
		sort_index = numpy.argsort(predict_rouge)[::-1]
		sort_index2 = numpy.argsort(important)[::-1]
		'''
		for j in range(10):
			tmp_summary.append(vec[sort_index[j]].sentence)
			tmp_name.append('PythonROUGE/'+i.name+'/'+str(j)+'.txt')
		'''
		lex_summary.append(vec[sort_index2[0]].sentence)
		tmp_summary.append(vec[sort_index[0]].sentence)
		produce_name.append('PythonROUGE/'+i.name+'/'+str(j)+'.txt')
		lex_name.append('PythonROUGE/'+i.name+'/'+str(j)+'_lex'+'.txt')
		produce_summary.append(tmp_summary)
	#print(standard_name)
	recall,precision,F_measure = summary_eval(standard_summary,standard_name,produce_summary,produce_name)
	print('recall:')
	print(recall)
	print('precision:')
	print(precision)
	print('F_measure:')
	print(F_measure)
	recall,precision,F_measure = summary_eval(standard_summary,standard_name,lex_summary,lex_name)
	print('lex_recall:')
        print(recall)
        print('lex_precision:')
        print(precision)
        print('lex_F_measure:')
        print(F_measure)
	return f_clf
	
def main():
	f_clf = bc3_eval()
	print('over')
	title = 'david_chalmers_how_do_you_explain_consciousness'
	#process_document(title,sen2=1,exist=0)
	sample_vector = []
	vec,important,title_sim = process_document(title,sen2=1)
        topic_similarity = lda_process(vec)
	maxs = 0
	maxi = 0
	phrase_list,vec_phrase_index = phrase_extraction(vec)
        word_list = word_score_calculate(phrase_list)
        vec_phrase_score = important_word(vec,word_list,phrase_list,vec_phrase_index)
	cue_word_score = cue_word(vec)
	#print(important)
	for j in range(len(vec)):
                        tmp = f_clf.predict([important[j],title_sim[j],topic_similarity[j],vec_phrase_score[j],cue_word_score[j]])
			if(tmp>=maxs):
				maxs = tmp
				maxi = j
	print(vec[maxi].sentence)
if __name__ == '__main__':
	main()
