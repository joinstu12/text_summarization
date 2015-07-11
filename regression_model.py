from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy, math
def feedForwardConstruct(numNeuron):
	n = FeedForwardNetwork()
	inLayer = LinearLayer(numNeuron[0])
	hiddenLayer = []
	for i in range(len(numNeuron)-2):
		hiddenLayer.append(SigmoidLayer(numNeuron[i+1]))
	outLayer = LinearLayer(numNeuron[-1])
	n.addInputModule(inLayer)
	for i in hiddenLayer:
		n.addModule(i)
	n.addOutputModule(outLayer)
	in_to_hidden = FullConnection(inLayer, hiddenLayer[0])
	hidden_to_hidden = []
	for i in range(len(hiddenLayer)-1):
		hidden_to_hidden.append(FullConnection(hiddenLayer[i],hiddenLayer[i+1]))
	hidden_to_out = FullConnection(hiddenLayer[-1],outLayer)
	n.addConnection(in_to_hidden)
	for i in hidden_to_hidden:
		n.addConnection(i)
	n.addConnection(hidden_to_out)
	n.sortModules()
	return n
num = [100,80,60,30,1]
network = feedForwardConstruct(num)
ds = SupervisedDataSet(100, 1)
for i in range(200):
	ds.addSample(numpy.ones(100),1)
trainer = BackpropTrainer(network, ds, verbose = True)
trainer.trainUntilConvergence(maxEpochs = 100)
