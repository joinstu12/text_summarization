import sys
sys.path.append('/home/kk/thesis/Text_Summarization/PythonROUGE')
import PythonROUGE
def save_summary_file(file,name,which):
	f = open(name,'w')
	#print(file)
	if(which == 0):
		#print(file)
		f.write(file.encode('utf-8')+'\n')
		'''
		for i in file:
			for j in i:
				f.write(j.encode('utf-8')+'\n')
		'''
	else:
		for i in file:
			#print name
			f.write(i.encode('utf-8')+'\n')
	f.close()
def summary_eval(standard_summary,standard_name,produce_summary,produce_name):
	#print(standard_name)
	for i in range(len(standard_name)):
		for j in range(len(standard_name[i])):
			save_summary_file(standard_summary[i][j],standard_name[i][j],0)
	#save_summary_file(standard_summary,standard_name)
	for i in range(len(produce_name)):
		save_summary_file(produce_summary[i],produce_name[i],1)
	#save_summary_file(produce_summary,produce_name)
	#print(produce_name)
	#print(standard_name)
	recall,precision,F_measure = PythonROUGE.PythonROUGE(produce_name,standard_name,ngram_order=1)
	#print(recall,precision,F_measure)
	#print('recall:')
	#print(recall)
	#print('precision:')
	#print(precision)
	#print('F_measure:')
	#print(F_measure)
	return recall,precision,F_measure
#s = [['Fuck the world']]
#p = [['Hey,fuck world']]
#summary_eval(s,[['PythonROUGE/st']],p,['PythonROUGE/pro'])
