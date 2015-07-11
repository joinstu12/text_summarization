from pyrouge import Rouge155
import os
class rouge(object):
	def __init__(self,evaluateSet,produceSet):
		self.evaluateSet = evaluateSet
		self.produceSet = produceSet
	def eval(self):
		r = Rouge155()
		os.mkdir('rouge_model')
		os.mkdir('rouge_system')
		self.write_file('rouge_model','rouge_system')
		r.system_dir = 'rouge_system'
		r.model_dir = 'rouge_model'
		r.system_filename_pattern = 'system.(\d+).txt'
		r.model_filename_pattern = 'model.[A-Z].#ID#.txt'

		output = r.convert_and_evaluate()
		#print(output)
		output_dict = r.output_to_dict(output)
		#print output_dict
		def remove_dir(top):
			for root, dirs, files in os.walk(top, topdown=False):
                		for name in files:
                        		os.remove(os.path.join(root, name))
                		for name in dirs:
                        		os.rmdir(os.path.join(root, name))
                	os.rmdir(top)
		remove_dir('rouge_model')
		remove_dir('rouge_system')
		return output_dict
	def write_file(self,model_root,system_root):
		def count_to_alphabet(count):
			if count == 0:
				return 'A'
			if count == 1:
				return 'B'
			if count == 2:
				return 'C'
		def int_to_digit(integer,digit):
			tmp = str(integer)
			if len(tmp) < digit:
				for i in range(digit-len(tmp)):
					tmp = '0' + tmp
			return tmp
		for mail_index in range(len(self.evaluateSet)):
			count = 0
			for annotator in self.evaluateSet[mail_index].annotators:
				f = open(model_root+'/model.'+count_to_alphabet(count)+'.' + int_to_digit(mail_index,3)+'.txt','w')
				for i in annotator:
					for j in self.evaluateSet[mail_index].text:
						if j.index == i:
							f.write(j.sentence + '\n')
							break
				f.close()
				count += 1
			f = open(system_root + '/system.' + int_to_digit(mail_index,3)+'.txt','w')
			for index in self.produceSet[mail_index]:
				for i in self.evaluateSet[mail_index].text:
					if i.index == index:
						f.write(i.sentence+'\n')
						break
			f.close()
