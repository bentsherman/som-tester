import heapq
import math
import numpy as np
import random
import sys



class GrowingNeuralGas(object):
	w = None
	C = None
	E = None

	def __init__(self, d):
		N = 2
		self.w = np.random.randn(N, d)
		self.C = np.zeros((N, N))
		self.E = np.zeros((N, 1))

	def fit(self, X, N_max):
		lmbda = 100
		epsilon_b = 0.2
		epsilon_n = 0.006
		alpha = 0.5
		a_max = 50
		d = 0.995

		for t in xrange(sys.maxint):
			N = self.w.shape[0]

			# sample x from X
			x = X[random.randint(0, X.shape[0] - 1)]

			# determine the first and second nearest units
			distances = [self.dist(x, self.w[i]) for i in xrange(N)]
			nearest = heapq.nsmallest(2, distances)
			s1 = distances.index(nearest[0])
			s2 = distances.index(nearest[1])

			# increment age of all edges from s1
			for i in xrange(N):
				if s1 != i and self.C[s1, i] > 0:
					self.C[s1, i] -= 1
					self.C[i, s1] -= 1

			# update error accumulator
			self.E[s1] += distances[s1]

			# update s1 and its neighbors towards x
			self.w[s1] += epsilon_b * (x - self.w[s1])

			for i in xrange(N):
				if s1 != i and self.C[s1, i] > 0:
					self.w[i] += epsilon_n * (x - self.w[i])

			# reset age of edge (s1, s2)
			self.C[s1, s2] = a_max
			self.C[s2, s1] = a_max

			# remove units which have no edges
			i = 0
			while i < self.w.shape[0]:
				if np.sum(self.C[i]) == 0:
					self.w = np.delete(self.w, i, axis=0)
					self.C = np.delete(self.C, i, axis=0)
					self.C = np.delete(self.C, i, axis=1)
					self.E = np.delete(self.E, i, axis=0)
				else:
					i += 1

			# insert a new unit periodically
			if t % lmbda == 0:
				# determine unit with maximum accumulated error
				q = max(xrange(len(self.E)), key=self.E.__getitem__)

				# determine neighbor with maximum accumulated error
				f = 0
				for i in xrange(self.w.shape[0]):
					if self.C[q, i] > 0 and self.E[f] < self.E[i]:
						f = i

				# insert new unit
				self.w = np.append(self.w, [(self.w[q] + self.w[f]) / 2], axis=0)
				self.C = np.append(self.C, np.zeros((1, self.C.shape[1])), axis=0)
				self.C = np.append(self.C, np.zeros((self.C.shape[0], 1)), axis=1)
				self.E = np.append(self.E, np.zeros((1, 1)), axis=0)

				r = -1

				# update edges
				self.C[q, f] = self.C[f, q] = 0
				self.C[r, q] = self.C[q, r] = a_max
				self.C[r, f] = self.C[f, r] = a_max

				# update accumulated errors
				self.E[q] *= alpha
				self.E[f] *= alpha
				self.E[r] = self.E[q]

			# decrease all errors
			self.E *= d

			# stop if network reaches maximum size
			if self.w.shape[0] == N_max:
				break

	def dist(self, v1, v2):
		return np.linalg.norm(v1 - v2)

	def dist_matrix(self):
		N = self.w.shape[0]
		D = np.zeros((N, N))

		for i in xrange(N):
			for j in xrange(N):
				D[i, j] = self.dist(self.w[i], self.w[j])

		return D

	def _label_node(self, labels, i, label):
		labels[i] = label
		for j in xrange(len(labels)):
			if labels[j] == 0 and self.C[i, j] > 0:
				self._label_node(labels, j, label)

	def clusters(self):
		labels = [0 for i in xrange(self.w.shape[0])]
		num_clusters = 0

		while labels.count(0) > 0:
			num_clusters += 1
			seed = labels.index(0)

			self._label_node(labels, seed, num_clusters)

		return num_clusters
