import os
import subprocess
class rouge(object):
	def __init__(self,evaluateSet,produceSet):
		self.write_setting(evaluateSet,produceSet)
	def write_setting(self,evaluateSet,produceSet):
		f = open('RELEASE-1.5.5/settings.xml','w')
        	f.write('<ROUGE_EVAL version="1.55">' + '\n')
        	f.write('\n')
        	count = 0
		os.mkdir('RELEASE-1.5.5/rouge')
        	for mail_index in range(len(evaluateSet)):
                	f.write('<EVAL ID="TASK_' + str(count) + '">' + '\n')
			os.mkdir('RELEASE-1.5.5/rouge/task' + str(count) + '_model')
			os.mkdir('RELEASE-1.5.5/rouge/task' + str(count) + '_peer')
			f.write('<MODEL-ROOT> RELEASE-1.5.5/rouge/task'+ str(count) + '_model'+'</MODEL-ROOT>' + '\n' \
				 + '<PEER-ROOT> RELEASE-1.5.5/rouge/task'+ str(count) +'_peer' +' </PEER-ROOT>' + '\n')
			f.write('<INPUT-FORMAT TYPE="SEE">  </INPUT-FORMAT>' + '\n')
			an_count = 1
			model_path = 'RELEASE-1.5.5/rouge/task'+ str(count) + '_model/'
			peer_path = 'RELEASE-1.5.5/rouge/task'+ str(count) +'_peer/'
			f.write('<MODELS>' + '\n')
			for annotator in evaluateSet[mail_index].annotators: #model_summary
				f.write('<M ID="' + str(an_count) +'">'  + 'human' + str(an_count) + '_doc' + str(mail_index) \
				 + '.html</M>' + '\n')
				model_summary = []
				for i in annotator:
					for j in evaluateSet[mail_index].text:
						if j.index == i:
							model_summary.append(j.sentence)
							break
				self.write_model(model_summary,mail_index,an_count,model_path)
				an_count += 1
			f.write('</MODELS>' + '\n')
			f.write('<PEERS>' + '\n')
			system_summary = []
			for index in produceSet[mail_index]:
				for i in evaluateSet[mail_index].text:
					if i.index == index:
						system_summary.append(i.sentence)
						break
			f.write('<P ID="1">' +'system' + '1' + '_doc' + str(mail_index) + '.html</P>' + '\n')
			f.write('</PEERS>' + '\n')
			self.write_system(system_summary,mail_index,1,peer_path)
                	f.write('</EVAL>' + '\n')
			count += 1
		f.write('</ROUGE_EVAL>' + '\n')
		print os.system('~/thesis/Text_Summarization/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ~/thesis/Text_Summarization/RELEASE-1.5.5/data -f A -a -x -s -m -2 -4 -u RELEASE-1.5.5/settings.xml > test.txt')
		#self.remove_dir('RELEASE-1.5.5/rouge')
	def remove_dir(self,top):
		for root, dirs, files in os.walk(top, topdown=False):
    			for name in files:
        			os.remove(os.path.join(root, name))
    			for name in dirs:
        			os.rmdir(os.path.join(root, name))
		os.rmdir(top)
	def write_model(self,summary,set_id,human_id,path):
		filename = path + 'human' + str(human_id) + '_doc' + str(set_id) + '.html'
                f = open(filename,'w')
                f.write('<html>'+'\n')
                f.write('<head><title>' + filename + '</title> </head>')
                f.write('<body bgcolor="white">')
                for i in range(len(summary)):
                        f.write('<a name="'+ str(i+1) +'">[' + str(i+1) +']</a>' \
                                + '<a href="#' + str(i+1) + '" id=' + str(i+1) \
                                + '>' + summary[i] + '</a>' + '\n')
                f.write('</html>')

	def write_system(self,summary,set_id,system_id,path):
		filename = path + 'system' + str(system_id) + '_doc' + str(set_id) + '.html'
		f = open(filename,'w')
		f.write('<html>'+'\n')
		f.write('<head><title>' + filename + '</title> </head>')
		f.write('<body bgcolor="white">')
		for i in range(len(summary)):
			f.write('<a name="'+ str(i+1) +'">[' + str(i+1) +']</a>' \
				+ '<a href="#' + str(i+1) + '" id=' + str(i+1) \
				+ '>' + summary[i] + '</a>' + '\n')
		f.write('</html>')
