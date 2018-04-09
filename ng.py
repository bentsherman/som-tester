import math
import numpy as np
import random



class NeuralGas(object):
	w = None

	def __init__(self, N, d):
		self.w = np.random.randn(N, d)

	def fit(self, X, num_iterations):
		N = self.w.shape[0]

		for t in xrange(num_iterations):
			# update hyperparameters
			lmbda = 10.0 - (10.0 - 1.0) * t / num_iterations
			epsilon = 0.05 - (0.05 - 0.001) * t / num_iterations

			# sample x from X
			x = X[random.randint(0, X.shape[0] - 1)]

			# compute distance of each weight from x
			d = [self.dist(x, self.w[i]) for i in xrange(N)]

			d_min = min(d)
			d_max = max(d)
			k = [(N - 1) * (d_i - d_min) / (d_max - d_min) for d_i in d]

			# update each weight according to k
			for i in xrange(N):
				self.w[i] += epsilon * math.exp(-k[i] / lmbda) * (x - self.w[i])

	def dist(self, v1, v2):
		return np.linalg.norm(v1 - v2)

	def dist_matrix(self):
		N = self.w.shape[0]
		D = np.zeros((N, N))

		for i in xrange(N):
			for j in xrange(N):
				D[i, j] = self.dist(self.w[i], self.w[j])

		return D
