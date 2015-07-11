import numpy
def lexrank(similarity_matrix,n_in,converge,dampling_factor=0.15):
	importance = numpy.ones(n_in)
	u = dampling_factor*(1.0/n_in)*numpy.ones((n_in,n_in))
	similarity = numpy.asarray(similarity_matrix)
	for i in range(n_in):
		#print(numpy.sum(similarity[i]))
		tmp_sum = numpy.sum(similarity[i])
		if(tmp_sum == 0):
			similarity[i] += 1/len(similarity[i])
			continue
		similarity[i] = similarity[i]/tmp_sum
	#print(len(similarity[0]))
	next_importance = numpy.dot(((u + (1-dampling_factor)*similarity).transpose()),importance)
	#print(u + (1-dampling_factor)*similarity)
	#print(len(next_importance))
	while(numpy.linalg.norm((importance-next_importance))>converge):
		#print(numpy.linalg.norm(next_importance))
		importance = next_importance
		next_importance = numpy.dot(((u + (1-dampling_factor)*similarity).transpose()),importance)
	return next_importance
