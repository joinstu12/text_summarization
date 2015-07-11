import random
import math
def divide_data(numData,rateTrain,rateValid,mails):
	index = range(numData)
	random.shuffle(index)
	numTrain = int(math.floor(numData*rateTrain))
	numValid = int(math.floor(numData*rateValid))
	trainIndex = index[0:numTrain]
	validIndex = index[numTrain:numTrain+numValid]
	testIndex = index[(numTrain+numValid):numData]
	trainSet = []
	testSet = []
	validSet = []
	for i in trainIndex:
		trainSet.append(mails[i])
	for i in testIndex:
		testSet.append(mails[i])
	for i in validIndex:
		validSet.append(mails[i])
	return (trainSet,validSet,testSet)
def weightRecall(evaluateSet,produceSet,write_folder = None):
	divisor = 0
	dividend = 0
	for i in range(len(evaluateSet)):
		if(write_folder is not None):
			f = open(write_folder+"/_"+str(i))
		for j in evaluateSet[i].text:
			if j.score > 0.0:
				divisor += j.score
			if j.index in produceSet[i]:
				dividend += j.score
				if write_folder is not None:
					f.write(j.sentence + '\n')
		if write_folder is not None:
			f.close()
	return dividend/divisor
