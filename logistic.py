import theano
import theano.tensor as T
import time
import os
from paragraph_vector_feature import readVectors,assignVectors
from xmlparser_multiple import parse_bc3
import numpy
from random import shuffle
import sys
from evaluation import divide_data,weightRecall
class logisticRegression(object):
	def __init__(self,input,n_in,n_out):
		self.W = theano.shared(value=numpy.zeros((n_in, n_out),dtype=theano.config.floatX),name='W',borrow=True)
		self.b = theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1) #argmax:return the class label that the sample most probabily belongs to
		self.params = [self.W, self.b]
	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.y_pred.type))
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
            		raise NotImplementedError()
def shared_dataset(data_x,data_y, borrow=True):
	shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')
def process_dataset(train_data,test_data,valid_data):
	train_set_x,train_set_y = shared_dataset(train_data[0],train_data[1])
	test_set_x,test_set_y = shared_dataset(test_data[0],test_data[1])
	valid_set_x,valid_set_y = shared_dataset(valid_data[0],valid_data[1])
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]	
	return rval
class data(object):
	def __init__(self,label,feature):
		self.label = label
		self.feature = feature
	def get_label(self):
		return self.label
	def get_feature(self):
		return self.feature
class target(object):
	def __init__(self,filename):
		self.data = []
		self.read_file(filename)
	def read_file(self,filename):
		f = open(filename,'r')
		for line in f.readlines():
			tmp = line.split()[1:]
			feature = []
			name = line.split()[0]
			for i in tmp:
				feature.append(float(i))
                        self.data.append(data(name,feature))
class dataset(object):
	def __init__(self,filename):
		self.data = []
		self.read_file(filename)
	def read_file(self,filename):
		f = open(filename,'r')
		for line in f.readlines():
			tmp = line.split()[1:]
			feature = []
			if line.split()[0] == '1':
				label = 1
			else:
				label = 0
			for i in tmp:
				feature.append(float(i.split(':')[1]))
			self.data.append(data(label,feature))
	def get_data(self):
		return self.data
	def get_train(self):
		return self.train
	def get_valid(self):
		return self.valid
	def get_test(self):
		return self.test
	def shuffle_data(self):
		shuffle(self.data)
	def split_data(self,train_rate,valid_rate):
		self.train = self.data[0:int((len(self.data)*train_rate))]
		self.valid = self.data[int((len(self.data)*train_rate)):int(((len(self.data)*train_rate) + len(self.data)*valid_rate))]
		self.test = self.data[int((len(self.data)*train_rate) + len(self.data)*valid_rate):]
def load_data(train,target_file):
	train_data = [[],[]]
	test_data = [[],[]]
	valid_data = [[],[]]
	f = open(train,'r')
	data_set = []
	data = dataset(train)
	target_data = target(target_file)
	data.shuffle_data()
	data.split_data(0.6,0.2)
	train_set = data.get_train()
	test_set = data.get_test()
	valid_set = data.get_valid()
	target_set = target_data.data
	def object_to_vector(data):
		array = [[],[]]
		for i in data:
			array[0].append(i.feature)
			array[1].append(i.label)
		return array
	train_data = object_to_vector(train_set)
	test_data = object_to_vector(test_set)
	valid_data = object_to_vector(valid_set)
	target_data = object_to_vector(target_set)
	print len(target_data[0])
	'''
	train = 1000
	test = 500
	vectors = readVectors('bc3_vector_with_subject')
	corpus = 'bc3/bc3corpus.1.0/corpus.xml'
	annotation = 'bc3/bc3corpus.1.0/annotation.xml'
	mails = parse_bc3(corpus,annotation)
	assignVectors(mails,vectors)
	count = 0
	
	trainSet,validSet,testSet = divide_data(len(mails),0.6,0.2,mails)
	def assignBinaryScore(dataset,output):
		for i in dataset:
			for j in i.text:
				if j.score>=0.32 and j.score<0.65:
					score = 1
				elif j.score >= 0.65 and j.score<0.99:
					score = 2
				elif j.score == 0:
					score = 0
				else:
					score = 3
				subject = i.subject_feature[int(j.index.split('.')[0]) - 1]
                                thread = i.thread_feature[int(j.index.split('.')[0]) - 1]
				output[0].append(j.vector + subject + thread)
				output[1].append(score)
	assignBinaryScore(trainSet,train_data)
	assignBinaryScore(validSet,valid_data)
	assignBinaryScore(testSet,test_data)
	'''
	return (process_dataset(train_data,test_data,valid_data),len(train_data[0][0]),target_data)

def sgd_optimization_mnist(dataset,target_data,learning_rate=0.13, n_epochs=10000,batch_size=250):
	datasets,length,target_data = load_data(dataset,target_data)
    	train_set_x, train_set_y = datasets[0]
    	valid_set_x, valid_set_y = datasets[1]
    	test_set_x, test_set_y = datasets[2]
	target_data_x = theano.shared(numpy.asarray(target_data[0],dtype=theano.config.floatX),borrow=True)
	target_data_y = target_data[1]
		
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
		
	index = T.lscalar()

	x = T.matrix('x')  # data, presented as rasterized images
    	y = T.ivector('y')
		
	classifier = logisticRegression(input=x, n_in=length, n_out=2)
	cost = classifier.negative_log_likelihood(y)
	test_model = theano.function(
		inputs=[index],
		outputs=classifier.errors(y),
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	prediction_model = theano.function(
		inputs=[],
		outputs = classifier.p_y_given_x,
		givens={
			x: target_data_x
		}
	)
	validate_model = theano.function(
        	inputs=[index],
        	outputs=classifier.errors(y),
        	givens={
           		x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            		y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        	}
    	)
		
	g_W = T.grad(cost=cost, wrt=classifier.W)
    	g_b = T.grad(cost=cost, wrt=classifier.b)
		
	updates = [(classifier.W, classifier.W - learning_rate * g_W),(classifier.b, classifier.b - learning_rate * g_b)]

	train_model = theano.function(
        	inputs=[index],
        	outputs=cost,
        	updates=updates,
        	givens={
            		x: train_set_x[index * batch_size: (index + 1) * batch_size],
            		y: train_set_y[index * batch_size: (index + 1) * batch_size]
        	}
    	)

	print '... training the model'
	patience = 5000
	patience_increase = 2
	improvement_threshold = 0.995
	validation_frequency = min(n_train_batches, patience / 2)

	best_validation_loss = numpy.inf
    	test_score = 0.
    	start_time = time.clock()

    	done_looping = False
    	epoch = 0
	
	while (epoch < n_epochs) and (not done_looping):
        	epoch = epoch + 1
        	for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index) 
			iter = (epoch - 1) * n_train_batches + minibatch_index #the number of batches that had trained
			
			if (iter + 1) % validation_frequency == 0:
				validation_losses = [validate_model(i)for i in xrange(n_valid_batches)]
                		this_validation_loss = numpy.mean(validation_losses)
				print(
                    			'epoch %i, minibatch %i/%i, validation error %f %%' %
                    			(
                        			epoch,
                        			minibatch_index + 1,
                        			n_train_batches,
                        			this_validation_loss * 100.
                    			)
                		)
				if this_validation_loss < best_validation_loss:
					if this_validation_loss < best_validation_loss * improvement_threshold:
                        			patience = max(patience, iter * patience_increase)	
					best_validation_loss = this_validation_loss

					test_losses = [test_model(i)for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)

					print(
                        			(
                            				'     epoch %i, minibatch %i/%i, test error of'
                            				' best model %f %%'
                        			) %
                        			(
                            				epoch,
                            				minibatch_index + 1,
                            				n_train_batches,
                            				test_score * 100.
                        			)
                    			)
				
				if patience <= iter:
                			done_looping = True
                			break
	end_time = time.clock()
	print(
        	(
            		'Optimization complete with best validation score of %f %%,'
            		'with test performance %f %%'
        	)
        	% (best_validation_loss * 100., test_score * 100.)
    	)
	print 'The code run for %d epochs, with %f epochs/sec' % (
        	epoch, 1. * epoch / (end_time - start_time))
	print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
	return prediction_model()
if __name__ == '__main__':
    sgd_optimization_mnist('train.txt','bc3_sentiment_vectors.txt')
