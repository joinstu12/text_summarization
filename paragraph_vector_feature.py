from xmlparser_multiple import parse_bc3
class paragraphFeature():
	def __init__(self,name):
		self.name = name
	        self.feature = []
def readVectors(filename):
	f = open(filename)
	mail_array = []
	now = -1
	for i in f.readlines():
		tmp = i.strip().split()
		if tmp[0][5:len(tmp[0])].split('_')[0] != now:
			mail_array .append([])
			now = tmp[0][5:len(tmp[0])].split('_')[0]
		mail_array[-1].append(paragraphFeature(tmp[0]))
		for i in tmp[1:len(tmp)]:
			i = float(i)
		mail_array[-1][-1].feature = tmp[1:len(tmp)]
	return mail_array
def assignVectors(mails,vectors):
	#print(len(vectors))
	for i in range(len(mails)):
		for j in vectors[i]:
			#print j.name.split('_')[-1][0:6]
			if j.name.split('_')[-1][0:7] == 'subject':
				mails[i].subject_feature.append(j.feature)
				continue
			if j.name.split('_')[-1][0:6] == 'thread':
				#print(j.name.split('_')[-1][0:6])
				mails[i].thread_feature.append(j.feature)
				continue
			if len(j.name.split('_')) == 2:
				mails[i].vector = j.feature
			for k in mails[i].text:
				if(k.index == j.name.split('_')[-1]):
					k.vector = j.feature
					break
			
	'''
	for i in range(len(mails)):
		#print len(mails[i].text)
		#print len(vectors[i])
		for j in range(len(vectors[i])):
			if j <len(mails[i].text):
				mails[i].text[j].vector = vectors[i][j].feature
			elif j == (len(vectors[i])-1):
				mails[i].vector = vectors[i][len(vectors[i])-1].feature
			else:
				mails[i].subject_feature.append(vectors[i][j].feature)
	
		
		for j in range(len(mails[i].text)):
			mails[i].text[j].vector = vectors[i][j].feature
		mails[i].vector = vectors[i][len(vectors[i])-1].feature
		
	'''
'''
vectors = readVectors('bc3_vector_with_subject')
for i in vectors:
	for j in i:
		print j.name
corpus = 'bc3/bc3corpus.1.0/corpus.xml'
annotation = 'bc3/bc3corpus.1.0/annotation.xml'
mails = parse_bc3(corpus,annotation)
assignVectors(mails,vectors)

for i in mails:
	for j in i.text:
		print j.index.split('.')
'''
