import numpy as np
import random



class SOFM(object):
	w = None

	def __init__(self, N, d):
		self.w = np.random.randn(N, d)

	def fit(self, X, num_iterations):
		N = self.w.shape[0]

		for t in xrange(0, num_iterations):
			# update hyperparameters
			radius = N / pow((t + 1.0) / 1000, 2)
			alpha = pow(0.999, t)

			# sample x from X
			x = X[random.randint(0, X.shape[0] - 1)]

			# find the nearest weight w_c to x
			d = [self.dist(x, self.w[i]) for i in xrange(N)]
			c = min(xrange(len(d)), key=d.__getitem__)

			# update each weight in the neighborhood of w_c
			for i in xrange(N):
				if abs(i - c) < radius:
					self.w[i] += alpha * (x - self.w[i])

	def dist(self, v1, v2):
		return np.linalg.norm(v1 - v2)

	def dist_matrix(self):
		N = self.w.shape[0]
		D = np.zeros((N, N))

		for i in xrange(N):
			for j in xrange(N):
				D[i, j] = self.dist(self.w[i], self.w[j])

		return D
