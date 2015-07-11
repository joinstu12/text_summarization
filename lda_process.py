import gensim
import numpy
def lda_process(vec):
	cor = []
        global_cor = []
        for i in vec:
                cor.append(i.remove_stop_ver)
                global_cor.extend(i.remove_stop_ver)
        dictionary = gensim.corpora.Dictionary(cor)
        id2word = {}
        for word in dictionary.token2id:
                id2word[dictionary.token2id[word]] = word
        tfidf = gensim.models.tfidfmodel.TfidfModel([dictionary.doc2bow(text) for text in cor])
        corpus = tfidf[[dictionary.doc2bow(text) for text in cor]]
        global_cor = tfidf[dictionary.doc2bow(global_cor)]
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=5)
        topic_distribution = []
        for i in lda[corpus]:
                tmp_distribution = []
                for j in i:
                        tmp_distribution.append(j[1])
                topic_distribution.append(numpy.asarray(tmp_distribution))
        #print(topic_distribution)
	#print('end')
        #print(lda[global_cor])
	index = 0
        global_distribution = []
        for i in lda[global_cor]:
		if(i[0]>index):
			global_distribution.append(0)
			index += 1
		index += 1
                global_distribution.append(i[1])
        global_distribution = numpy.asarray(global_distribution)
        #print(global_distribution)
	#print('local')
	#print(len(topic_distribution[0]))
	#print('global')
	#print(len(global_distribution))
	'''
	if(len(topic_distribution[0])!=len(global_distribution)):
		print(lda[global_cor])
	'''
        topic_similarity = []
        for i in topic_distribution:
                topic_similarity.append(numpy.dot(i,global_distribution.T)/numpy.linalg.norm(i)/numpy.linalg.norm(global_distribution))
        #print(topic_similarity)
	'''
        for i in range(0, lda.num_topics):
                print lda.print_topic(i)
	'''
	topic_similarity = (topic_similarity-min(topic_similarity))/(max(topic_similarity)-min(topic_similarity))
	return topic_similarity
