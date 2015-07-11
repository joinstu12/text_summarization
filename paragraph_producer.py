from xmlparser_multiple import parse_file
from xmlparser_multiple import parse_anno
def main():
	corpus = 'bc3/bc3corpus.1.0/corpus.xml'
	annotation = 'bc3/bc3corpus.1.0/annotation.xml'
	output = 'bc3_paragraph_with_subject'
	mails = parse_anno(annotation,parse_file(corpus))
	output = open(output,'w')
	for i in xrange(len(mails)):
		tmp_doc = ''
		tmp_thread = ''
		thread = '1'
		for j in mails[i].text:
			if j.index.split('.')[0] != thread:
				output.write('_mail' + str(i)+'_thread'+ str(thread)  + ' ' + tmp_thread + '\n')
				tmp_thread = ''
				thread = j.index.split('.')[0]
			output.write('_mail' + str(i) + '_' + j.index + ' ' + j.sentence + '\n')
                        tmp_doc += j.sentence + ' '
			tmp_thread += j.sentence
		output.write('_mail' + str(i)+'_thread'+ str(thread)  + ' ' + tmp_thread + '\n')
		for j in range(len(mails[i].subject)):
			output.write('_mail' + str(i) + '_' + 'subject' + str(j) +' ' + mails[i].subject[j] + '\n')
		output.write('_mail' + str(i) + ' ' + tmp_doc + '\n')
if __name__ == '__main__':
	main()
