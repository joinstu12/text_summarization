from logistic_sgd import logisticRegression, load_data,process_test_data
from evaluation import weightRecall
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,activation=T.tanh):
		self.input = input
		if W is None:
            		W_values = numpy.asarray(
                		rng.uniform(
                    			low=-numpy.sqrt(6. / (n_in + n_out)),
                    			high=numpy.sqrt(6. / (n_in + n_out)),
                    			size=(n_in, n_out)
                		),
                		dtype=theano.config.floatX
            		)
			if activation == theano.tensor.nnet.sigmoid:
                		W_values *= 4
			
			W = theano.shared(value=W_values, name='W', borrow=True)
		
		if b is None:
           		b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            		b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
            		lin_output if activation is None
            		else activation(lin_output)
        	)

		self.params = [self.W, self.b]

class MLP(object):
	def __init__(self, rng, input, n_in, hidden_layers_sizes, n_out):
		'''
		self.hiddenLayer = HiddenLayer(
            		rng=rng,
            		input=input,
            		n_in=n_in,
           		n_out=n,
            		activation=T.tanh
        	)
		'''
		self.params = []
		self.sigmoid_layers = []
		self.n_layers = len(hidden_layers_sizes)
		for i in xrange(self.n_layers):
			if i == 0:
                		input_size = n_in
            		else:
                		input_size = hidden_layers_sizes[i - 1]

			if i == 0:
                		layer_input = input
            		else:
                		layer_input = self.sigmoid_layers[-1].output
			sigmoid_layer = HiddenLayer(rng=rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
			self.sigmoid_layers.append(sigmoid_layer)
			self.params.extend(sigmoid_layer.params)	
	
        	self.logRegressionLayer = logisticRegression(
							input=self.sigmoid_layers[-1].output,
            						n_in=hidden_layers_sizes[-1],
            						n_out=n_out
        					)
			
		self.L1 = abs(self.logRegressionLayer.W).sum()
		for i in self.sigmoid_layers:
			self.L1 += abs(i.W).sum()
		'''	
		self.L1 = (
            		abs(self.hiddenLayer.W).sum()
            		+ abs(self.logRegressionLayer.W).sum()
        	)
		'''
		self.L2_sqr = abs(self.logRegressionLayer.W ** 2).sum()
                for i in self.sigmoid_layers:
                        self.L2_sqr += abs(i.W ** 2).sum() 
		'''
        	self.L2_sqr = (
            		(self.hiddenLayer.W ** 2).sum()
            		+ (self.logRegressionLayer.W ** 2).sum()
        	)
		'''
        	self.negative_log_likelihood = (
            		self.logRegressionLayer.negative_log_likelihood
        	)

        	self.errors = self.logRegressionLayer.errors
		self.params.extend(self.logRegressionLayer.params)
        	#self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(dataset,learning_rate=0.1, L1_reg=0.00, L2_reg=0.0001, n_epochs=2000, batch_size=200, hidden_layers_sizes=[350,230,150,80,30]):
	datasets,length,testSet = load_data(dataset)
    	train_set_x, train_set_y = datasets[0]
    	valid_set_x, valid_set_y = datasets[1]
    	test_set_x, test_set_y = datasets[2]

    	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    	######################
    	# BUILD ACTUAL MODEL #
    	######################
    	print '... building the model'

    	index = T.lscalar()  # index to a [mini]batch
    	x = T.matrix('x')  # the data is presented as rasterized images
    	y = T.ivector('y')  # the labels are presented as 1D vector of
                        	# [int] labels

    	rng = numpy.random.RandomState(1234)

    	classifier = MLP(
        	rng=rng,
        	input=x,
        	n_in=length,
        	hidden_layers_sizes=hidden_layers_sizes,
        	n_out=4
    	)

    	# start-snippet-4
    	# the cost we minimize during training is the negative log likelihood of
    	# the model plus the regularization terms (L1 and L2); cost is expressed
    	# here symbolically
    	cost = (
        	classifier.negative_log_likelihood(y)
        	+ L1_reg * classifier.L1
        	+ L2_reg * classifier.L2_sqr
    	)
    	# end-snippet-4

    	# compiling a Theano function that computes the mistakes that are made
    	# by the model on a minibatch
    	test_model = theano.function(
        	inputs=[index],
        	outputs=classifier.errors(y),
        	givens={
            		x: test_set_x[index * batch_size:(index + 1) * batch_size],
            		y: test_set_y[index * batch_size:(index + 1) * batch_size]
        	}
    	)

    	validate_model = theano.function(
        	inputs=[index],
        	outputs=classifier.errors(y),
        	givens={
            		x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            		y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        	}
    	)
	''' test momentum
    	# start-snippet-5
    	# compute the gradient of cost with respect to theta (sotred in params)
    	# the resulting gradients will be stored in a list gparams
    	gparams = [T.grad(cost, param) for param in classifier.params]

    	# specify how to update the parameters of the model as a list of
    	# (variable, update expression) pairs

    	# given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    	# same length, zip generates a list C of same size, where each element
    	# is a pair formed from the two lists :
    	#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    	updates = [
        	(param, param - learning_rate * gparam)
        	for param, gparam in zip(classifier.params, gparams)
    	]
	'''
	updates = []
	momentum = 0.9
	#param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
	for param in classifier.params:
		param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
		updates.append((param, param - learning_rate*param_update))
        	# Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        	updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
	#def gradient_updates_momentum(cost, params, learning_rate, momentum):
	#	assert momentum < 1 and momentum >= 0
		
    	# compiling a Theano function `train_model` that returns the cost, but
    	# in the same time updates the parameter of the model based on the rules
    	# defined in `updates`
	
    	train_model = theano.function(
        	inputs=[index],
        	outputs=cost,
        	updates=updates,
        	givens={
            		x: train_set_x[index * batch_size: (index + 1) * batch_size],
            		y: train_set_y[index * batch_size: (index + 1) * batch_size]
        	}
    	)
    	# end-snippet-5

    	###############
    	# TRAIN MODEL #
    	###############
    	print '... training'

    	# early-stopping parameters
    	patience = 10000  # look as this many examples regardless
    	patience_increase = 2  # wait this much longer when a new best is
                           # found
    	improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    	validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    	best_validation_loss = numpy.inf
    	best_iter = 0
    	test_score = 0.
    	start_time = time.clock()

    	epoch = 0
    	done_looping = False

    	while (epoch < n_epochs) and (not done_looping):
        	epoch = epoch + 1
        	for minibatch_index in xrange(n_train_batches):

            		minibatch_avg_cost = train_model(minibatch_index)
            		# iteration number
            		iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
                		# compute zero-one loss on validation set
                		validation_losses = [validate_model(i) for i
                                			in xrange(n_valid_batches)]
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

                		# if we got the best validation score until now
                		if this_validation_loss < best_validation_loss:
                    			#improve patience if loss improvement is good enough
                    			if (
                        			this_validation_loss < best_validation_loss *
                        			improvement_threshold
                    			):
                        			patience = max(patience, iter * patience_increase)

                    			best_validation_loss = this_validation_loss
                    			best_iter = iter

                    			# test it on the test set
                    			test_losses = [test_model(i) for i
                                   			in xrange(n_test_batches)]
                    			test_score = numpy.mean(test_losses)

                    			print(('     epoch %i, minibatch %i/%i, test error of '
                           			'best model %f %%') %
                          			(epoch, minibatch_index + 1, n_train_batches,
                           			test_score * 100.))

			if patience <= iter:
                		done_looping = True
                		break

	end_time = time.clock()
	print(('Optimization complete. Best validation score of %f %% '
        	'obtained at iteration %i, with test performance %f %%') %
        	(best_validation_loss * 100., best_iter + 1, test_score * 100.))
    	print >> sys.stderr, ('The code for file ' +
        		os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((end_time - start_time) / 60.))
	prediction_model = theano.function(
                inputs=[],
                outputs = classifier.logRegressionLayer.y_pred,
                givens={
                        x: test_set_x
                }
        )
	produceSet = process_test_data(prediction_model(),testSet)
        print weightRecall(testSet,produceSet)
        print produceSet
if __name__ == '__main__':
    test_mlp('123')
