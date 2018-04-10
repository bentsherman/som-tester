import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

import ng
import sofm


def load_ematrix(filename):
	print "Loading expression matrix..."

	# read expression matrix from file
	df = pd.read_csv(filename, sep="\t", index_col=0)

	# remove NAN columns
	df = df.dropna(axis=1, how="all")

	print "Loaded expression matrix", df.shape

	return df



def load_pairs(filename):
	pairs = pd.read_csv(filename, sep="\t", header=None)

	return pairs



def plot_sofm(X, N, num_iterations):
	model = sofm.SOFM(N, X.shape[1])

	# visualize distance matrix (initial)
	ax = plt.subplot(232)
	sns.heatmap(model.dist_matrix())

	# train model
	model.fit(X, num_iterations)

	# visualize final weights
	ax = plt.subplot(231)
	plt.plot(X[:, 0], X[:, 1], "ko", markersize=2)
	plt.plot(model.w[:, 0], model.w[:, 1], "gx-", markersize=5)

	# visualize distance matrix (final)
	ax = plt.subplot(233)
	sns.heatmap(model.dist_matrix())



def plot_ng(X, N, num_iterations):
	model = ng.NeuralGas(N, X.shape[1])

	# visualize distance matrix (initial)
	ax = plt.subplot(235)
	sns.heatmap(model.dist_matrix())

	# train model
	model.fit(X, num_iterations)

	# visualize final weights
	ax = plt.subplot(234)
	plt.plot(X[:, 0], X[:, 1], "ko", markersize=2)
	plt.plot(model.w[:, 0], model.w[:, 1], "go", markersize=10)

	# visualize distance matrix (final)
	ax = plt.subplot(236)
	sns.heatmap(model.dist_matrix())



if __name__ ==  "__main__":
	if len(sys.argv) != 5:
		print "usage: python som-tester.py [ematrix-file] [pairs-file] [N] [num-iter]"
		sys.exit(1)

	df = load_ematrix(sys.argv[1])
	pairs = load_pairs(sys.argv[2])
	N = int(sys.argv[3])
	NUM_ITERATIONS = int(sys.argv[4])

	OUTPUT_DIR = "plots-%02d-%05d" % (N, NUM_ITERATIONS)

	if not os.path.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

	plt.figure(figsize=(15, 10))

	for idx in pairs.index:
		i = pairs[0][idx]
		j = pairs[1][idx]

		print i, j

		# extract pairwise data
		X = df.iloc[[i, j]].dropna(axis=1, how="any")
		X = X.T.values

		# create plots of SOFM and NG
		plt.clf()

		plot_sofm(X, N, NUM_ITERATIONS)
		plot_ng(X, N, NUM_ITERATIONS)

		plt.savefig("%s/%06d_%06d.png" % (OUTPUT_DIR, i, j))
