"""
This file provides fucntions for shuffling, shifting and simulating neurons
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore



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
# test
#shift_neurons(dat['sresp'][:20])
  
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

# test
#shuffle_neurons(dat['sresp'][:20]).shape

def plot_carpet(data, nrange = np.arange(1100, 1400), title = None):

	plt.figure(figsize=(16, 8))
	ax = plt.subplot(111)
	plt.imshow(shift[nrange], vmax= 3, vmin=-1, aspect='auto', cmap='gray_r')
	ax.set(xlabel='timepoints', ylabel='neurons')
	plt.title(title)
	plt.show()

if __name__ == "__main__":

	## Test the functions
	# load functions
	dat = np.load('stringer_spontaneous.npy', allow_pickle=True).item()
	#dat_ori = np.load('stringer_orientations.npy', allow_pickle=True).item()
	
	# shift and shuffle
	shift = shift_neurons(dat['sresp'], chunk_size = dat['sresp'].shape[0]//100, shuffle = False)
	shuffle = shuffle_neurons(dat['sresp'])
	
	# plot
	shift = zscore(shift)
	shuffle = zscore(shuffle)
	
	plot_carpet(shift)
	plot_carpet(shuffle)


















