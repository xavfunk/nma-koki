"""
This file provides fucntions for shuffling, shifting and simulating neurons
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.optimize import curve_fit
from scipy.stats import halfnorm

## shifting and shuffling

def shift_neurons(neurons, shift_range = None, chunk_size = 1, shuffle = False):
	
	"""
	shifts neuronal time series, according to shift parameter

	inputs:
	neurons: array; neuronal time series of shape n_neurons x timepoints
	shift_range: int; range of shifting, list or tuple of length 2
	chink_size: specifies how many neurons are shifted together in one 'chunk'

	outputs:
	shifted_neurons: array: each neuron's index is randomized and their time series is shifted
	"""
	# define the shift range
	if shift_range is None:
		shift_range = (0, neurons.shape[1])

	# initialize result array
	shifted_neurons = np.zeros_like(neurons)

	# shuffle the neurons
	shuffled_neurons = neurons.copy()

	if shuffle == True:
		np.random.shuffle(shuffled_neurons)

	indeces = list(range(neurons.shape[0]))
	indeces = [indeces[i:i+chunk_size] for i in range(0,len(indeces),chunk_size)]

	# loop over neurons
	for index in indeces:

		try:
		  shift = np.random.randint(*shift_range)

		except TypeError:
		  shift = shift_range

		# shift neuron time series
		shifted_neurons[index] = np.roll(shuffled_neurons[index], shift)

	return shifted_neurons
  
def shuffle_neurons(neurons):
	
	"""
	shuffles neuronal time series
	inputs:
	neurons: array; neuronal time series of shape n_neurons x timepoints

	outputs:
	shuffled_neurons: array: each neuron's time series is randomized
	"""
	# copy neurons into result array, using copy and .T to shuffle the columns
	shuffled_neurons = neurons.copy().T

	# shuffle
	np.random.shuffle(shuffled_neurons)

	return shuffled_neurons.T

## simulations

def fit_halfnorm(k, s):
    
    '''halfnorm function, parameter s is the standard deviation'''
    
    return halfnorm.pdf(k, 0, s)

def simulate_neurons(neurons, plot_hist = False):
	
	"""
	Takes an array of neuronal activations and simulates neurons with similar activity distribution.
	Does not conserve activity timings. 
	The array of simulated neurons has the same shape as the input neurons. 
	
	inputs:
	neurons: array; neuronal time series of shape n_neurons x timepoints
	
	outputs:
	synthetic_neurons: array: synthetic data for each neuron
	params: array: array of halfnormal std
	"""
	
	# initialize result arrays
	synthetic_neurons = np.zeros_like(neurons)
	params = np.zeros(dat['sresp'].shape[0]) # will contain halfnormal stds of every neuron

	# loop over neurons
	for i, neuron in enumerate(neurons):
	
		# clean the neuron by removing zeros
		neuron_clean = neuron[neuron > 0]
		# keep track of number of zeros
		n_zeros = ((neuron > 0) == False).sum()

		# initialize bins for histogram
		bins = np.arange(200) - 0.5

		## get histogram
		if plot_hist is True:
			# with plt, plots the hgram
			entries, bin_edges, patches = plt.hist(neuron_clean, bins=bins, density=True, label='Data')
		else:
			# with numpy, not plotting
			entries, bin_edges = np.histogram(neuron_clean, bins=bins, density=True)

		# get bin middles for accurate x's    	
		bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])

		# Halfgaussian fit with curve_fit
		p0 = 10 # starting value 10 seemed reasonable and worked reliably
		parameters_n, cov_matrix_n = curve_fit(fit_halfnorm, bin_middles[1:], entries[1:], p0 = p0)

		# simulate with fitted parameters
		synthetic_neuron = halfnorm.rvs(scale = parameters_n, size=neurons.shape[1])

		# insert zeros
		idxs = np.random.choice(neurons.shape[1], int(n_zeros), replace=False)
		synthetic_neuron[idxs] = 0 

		# insert result
		synthetic_neurons[i] = synthetic_neuron
		params[i] = parameters_n

	return synthetic_neurons, params

## plotting

def plot_carpet(data, nrange = np.arange(1100, 1400), title = None):

	plt.figure(figsize=(16, 8))
	ax = plt.subplot(111)
	plt.imshow(shift[nrange], vmax= 3, vmin=-1, aspect='auto', cmap='gray_r')
	ax.set(xlabel='timepoints', ylabel='neurons')
	plt.title(title)
	plt.show()

if __name__ == "__main__":

	## Test the functions
	
	# load data
	dat = np.load('stringer_spontaneous.npy', allow_pickle=True).item()
	#dat_ori = np.load('stringer_orientations.npy', allow_pickle=True).item()
	
	# shift and shuffle
	shift = shift_neurons(dat['sresp'], chunk_size = dat['sresp'].shape[0]//100, shuffle = False)
	shuffle = shuffle_neurons(dat['sresp'])
	
	# plot_carpet
	shift = zscore(shift)
	shuffle = zscore(shuffle)
	
	plot_carpet(shift)
	plot_carpet(shuffle)
	
	
	# simulate
	synthetic_neurons, params = simulate_neurons(dat['sresp'][:20])
	
	i = 19
	scale = 1
	bins = np.arange(200) - 0.5
	plt.hist(synthetic_neurons[i]*scale, bins = bins, alpha = .5)
	plt.hist(dat['sresp'][i], bins = bins, alpha = .5)
	#plt.ylim(0,500)
	plt.show()












