import theano
import theano.tensor as T
import time
import os
from paragraph_vector_feature import readVectors,assignVectors
from xmlparser_multiple import parse_bc3
import numpy
import sys
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
def load_data(dataset):
	train_data = [[],[]]
	test_data = [[],[]]
	valid_data = [[],[]]
	train = 1000
	test = 500
	vectors = readVectors('bc3_vectors.txt')
	corpus = 'bc3/bc3corpus.1.0/corpus.xml'
	annotation = 'bc3/bc3corpus.1.0/annotation.xml'
	mails = parse_bc3(corpus,annotation)
	assignVectors(mails,vectors)
	count = 0
	for i in mails:
        	for j in i.text:
			if j.score > 0:
                        	score = 1
                        else:
                        	score = 0
			if count<train:
                		train_data[0].append(j.vector + i.vector)
				train_data[1].append(score)
			elif count>train and count<train+test:
				test_data[0].append(j.vector + i.vector)
                                test_data[1].append(score)
			else:
				valid_data[0].append(j.vector + i.vector)
                                valid_data[1].append(score)
			count += 1
	return (process_dataset(train_data,test_data,valid_data),len(train_data[0][0]))
def sgd_optimization_mnist(dataset,learning_rate=0.13, n_epochs=1000,batch_size=250):
	datasets,length = load_data(dataset)
    	train_set_x, train_set_y = datasets[0]
    	valid_set_x, valid_set_y = datasets[2]
    	test_set_x, test_set_y = datasets[1]
		
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
if __name__ == '__main__':
    sgd_optimization_mnist('123')
