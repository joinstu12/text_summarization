import numpy 
import theano
import theano.tensor as T
from collections import OrderedDict
import cPickle as pickle

class rnn(object):
	def __init__(self, nh, ni, no,L1_reg = 0.0,L2_reg = 0.0,write = False):
		self.L1_reg = float(L1_reg)
		self.L2_reg = float(L2_reg)
		self.w_i_to_h = theano.shared(name='w_i_to_h',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (ni, nh))
                                .astype(theano.config.floatX))
		self.w_h_to_h = theano.shared(name='w_h_to_h',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
		self.w_h_to_o = theano.shared(name='w_h_to_o',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh,no))
                                .astype(theano.config.floatX))
		self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        	self.b = theano.shared(name='b',
                               value=numpy.zeros(no,
                               dtype=theano.config.floatX))
        	self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
		self.params = [self.w_i_to_h, self.w_h_to_h, self.w_h_to_o,
                       		self.bh, self.b, self.h0]
		self.L1 = 0.0
		self.L1 += abs(self.w_i_to_h.sum())
		self.L1 += abs(self.w_h_to_h.sum())
		self.L1 += abs(self.w_h_to_o.sum())

		self.L2_sqr = 0.0
		self.L2_sqr += (self.w_i_to_h ** 2).sum()
		self.L2_sqr += (self.w_h_to_h ** 2).sum()
		self.L2_sqr += (self.w_h_to_o ** 2).sum()
		self.hidden_layer = []
		self.write = write
		def recurrence(x_t, h_tm1):
        		h_t = T.nnet.sigmoid(T.dot(x_t, self.w_i_to_h)
                		+ T.dot(h_tm1, self.w_h_to_h) + self.bh)
            		s_t = T.nnet.sigmoid(T.dot(h_t, self.w_h_to_o) + self.b)
			if self.write:
				self.hidden_layer.append(h_t)
            		return [h_t, s_t]
		x = T.fmatrix()
        	#x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        	y_sentence = T.fmatrix('y_sentence')  # labels
		[h, s], _ = theano.scan(fn=recurrence,
        	                        sequences=x,
                	                outputs_info=[self.h0, None],
                	                n_steps=x.shape[0])
		
		p_y_given_x_sentence = s#[:, 0, :]
        	y_pred = p_y_given_x_sentence#T.argmax(p_y_given_x_sentence, axis=1)
		lr = T.scalar('lr')

        	#sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                #	               [T.arange(x.shape[0]), y_sentence]) + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr
		sentence_nll = T.mean((y_pred - y_sentence) ** 2)
        	sentence_gradients = T.grad(sentence_nll, self.params)
        	sentence_updates = OrderedDict((p, p - lr*g)
                	                       for p, g in
                        	               zip(self.params, sentence_gradients))
	
		self.classify = theano.function(inputs=[x], outputs=y_pred)
		self.prob = theano.function(inputs=[x], outputs=p_y_given_x_sentence)
        	self.sentence_train = theano.function(inputs=[x, y_sentence, lr],
                                              	outputs=sentence_nll,
                                              	updates=sentence_updates)
	def get_hidden(self,x_t):
		def recurrence(x_t, h_tm1):
                        h_t = T.nnet.sigmoid(T.dot(x_t, self.w_i_to_h)
                                + T.dot(h_tm1, self.w_h_to_h) + self.bh)
                        s_t = T.nnet.sigmoid(T.dot(h_t, self.w_h_to_o) + self.b)
                        if self.write:
                                self.hidden_layer.append(h_t.eval())
                        return [h_t, s_t]
		h_t = self.h0
		for i in x_t:
			h_t = recurrence(i,h_t)[0]
	def flush_hidden(self):
		self.hidden_layer = []
	def get_weights(self):
		weights = [p.get_value() for p in self.params]
		return weights
	def set_weights(self,weights):
		i = iter(weights)
		
		for param in self.params:
			param.set_value(i.next())
	def save(self,filename):
		weights = self.get_weights()
		file = open(filename,'wb')
		pickle.dump(weights, file, protocol=pickle.HIGHEST_PROTOCOL)
		file.close()
	def load(self,filename):
		file = open(filename,'rb')
		weights = pickle.load(file)
		self.set_weights(weights)
		file.close()
