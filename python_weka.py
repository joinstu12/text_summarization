import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
import os
class python_weka(object):
	def __init__(self,input_x,input_y,labels):
		self.input_x = input_x
		self.input_y = input_y
		self.labels = labels
		
	def write_arff(self,filename,relation,train_or_predict,input_x,input_y = None):
		f = open(filename,'w')
		f.write('@relation ' + relation + '\n')
		for i in self.labels:
			train_or_predict += 1
			if train_or_predict == len(self.labels):
				break
			f.write('@attribute ' + i + ' ' + self.labels[i] + '\n')
		f.write('\n')
		f.write('@data' + '\n')
		for i in range(len(input_x)):
			for j in input_x[i]:
				f.write(str(j) + '  ' )
			if train_or_predict == 0:
				f.write(str(input_y[i]))
			else:
				f.write(str(0))
			f.write('\n')
		f.close()
	def train(self):
		filename = 'train.arff'
		self.write_arff(filename,'train',0,self.input_x,self.input_y)
		loader = Loader(classname="weka.core.converters.ArffLoader")
		data = loader.load_file(filename)
		data.class_is_last()
		self.cls = Classifier(classname="weka.classifiers.meta.Bagging", options=['-S','5'])
		self.cls.build_classifier(data)
		os.remove(filename)
	def predict(self,test_data):
		filename = 'test.arff'
		self.write_arff(filename,'test',0,test_data)
		loader = Loader(classname="weka.core.converters.ArffLoader")
		data = loader.load_file(filename)
		data.class_is_last()
		#evl = Evaluation(data)
		#evl.evaluate_model(self.cls,data)
		#data.set_class_label(data.numAttributes() - 1)
		#data.setClassIndex(data.numAttributes() - 1)
		result = []
		for index, inst in enumerate(data):
			pred = self.cls.classify_instance(inst)
			dist = self.cls.distribution_for_instance(inst)
			result.append(dist[0])
			#print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))
			#print str(index+1) + 'dist:'+ str(dist)
		os.remove(filename)
		return result
		
