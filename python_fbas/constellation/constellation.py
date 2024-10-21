import numpy as np 

def cluster(r):
	"""
	function to compute a cluster assignment 
    given desired cluster sizes of orgs
    """

	n = len(r)

	# r_array: 
	# 1st column: index of org 
	# 2nd column: required degree
	# 3rd column: size of cluster allocated to org 
	# 4th column: index of cluster allocated to org 
	# 5th column: degree actually achieved by org
	r_array = np.zeros([n, 5])
	r_array[:, 0] = range(n)
	r_array[:, 1] = r

	# sort required degree in descending order 
	r_array = r_array[r_array[:,1].argsort()[::-1]]

	# org counter 
	i = 0
	
	# cluster index counter 
	j = 0

	while i < n:

		# if there are leftover orgs
		if n - i < r_array[i, 1] + 1:
			
			if i > 0:
				
				# merge leftover orgs with previous cluster 
				r_array[i:, 2] = r_array[i-1, 2] + n - i
				r_array[i:, 3] = r_array[i-1, 3]

				# update previous cluster 
				prev_cluster = np.min(np.where(r_array[:, 3] == r_array[i-1, 3])[0])
				r_array[prev_cluster:i, 2] = r_array[i-1, 2] + n - i

				# update i and j
				i = n
				j = j + 1

			else: 
				raise Exception("Sorry, no valid clusters possible")

		else: 

			# form new cluster per the descending order clustering algorithm 
			r_array[i:i+int(r_array[i, 1])+1, 2] = r_array[i, 1] + 1 
			r_array[i:i+int(r_array[i, 1])+1, 3] = j

			# update i and j
			i = i + int(r_array[i, 1]) + 1
			j = j + 1

	# compute degree achieved by above clustering 
	r_array = degree_calculator(r_array)

	return r_array


""" function to compute degree achieved by an org 
given a clustering assignment """

def degree_calculator(r_array):

	# number of clusters 
	num_clusters = int(r_array[-1, 3]) + 1

	# number of orgs 
	n = len(r_array[:, 0]) 

	# compute degree achieved by each org 
	for i in range(n): 

		for j in range(num_clusters): 

			# count intra-cluster edges 
			if r_array[i, 3] == j: 

				r_array[i, 4] += (r_array[i, 2] - 1)

			# count inter-cluster edges
			else: 

				size_of_cluster_j = r_array[np.min(np.where(r_array[:, 3] == j)[0]), 2]

				size_of_cluster_i_is_in = r_array[i, 2]

				index_of_i_within_its_cluster = i - np.min(np.where(r_array[:, 3] == r_array[i, 3])[0])

				# 2 cases. 1st case if cluster j is bigger than i's cluster. 
				if size_of_cluster_j > size_of_cluster_i_is_in:

					r_array[i, 4] += np.floor(size_of_cluster_j / size_of_cluster_i_is_in)

					if index_of_i_within_its_cluster < size_of_cluster_j % size_of_cluster_i_is_in: 
				
						r_array[i, 4] += 1

				# 2nd case. cluster j is smaller than i's cluster. 
				else: 

					r_array[i, 4] += 1

	return r_array


def main(): 

	# iterations 
	num_iter = 50

	# step size parameter 
	eta = 0.5

	# number of orgs
	n = 30

	# thresholds 
	t = [20]*n 

	# required degree 
	d = [n + 1 - i for i in t]

	# required cluster size (minus one). r can be fractional. 
	r = [d[i] for i in range(n)]

	for i in range(num_iter):

		# convert fractional r values to integral 
		int_r = [int(r[i]) for i in range(n)]

		# perform clustering 
		r_array = cluster(int_r)

		# print results 
		print("Iteration: ", i)
		print("Clustering: format is (O, C_O, O's cluster size, O's cluster index, total degree achieved by O)")
		print(r_array)
		print(" ")

		# sort r_array by ascending order of org index
		r_array = r_array[r_array[:, 0].argsort()]

		""" incremental update of r.  
		note that we perform the update on the fractional r used 
		in the previous round before the rounding operation """
		r = [max(0, r[i] - eta*int(d[i] < r_array[i, 4] and r_array[i, 4] < 2*d[i]) 
			+ eta*int(d[i] > r_array[i, 4] or r_array[i, 4] >= 2*d[i])) for i in range(n)]			

		# alternative r update equation 
		# r = [max(0, int(r_array[i, 1] - eta*int(d[i] < r_array[i, 4] and r_array[i, 4] < 2*d[i])*(r_array[i, 4] - d[i]) 
			# + eta*int(d[i] > r_array[i, 4] or r_array[i, 4] >= 2*d[i])*abs(d[i] - r_array[i, 4]))) for i in range(n)]


if __name__ == '__main__':
	main()