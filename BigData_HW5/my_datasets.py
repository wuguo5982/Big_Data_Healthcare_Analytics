import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	newData = pd.read_csv(path)
	XX = newData.loc[:, 'X1':'X178'].values
	YY = (newData.loc[:,'y'] - 1).values
	if model_type == 'MLP':
		data = torch.from_numpy(XX.astype(np.float32))
		target = torch.from_numpy(YY)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = torch.from_numpy(XX.astype(np.float32))
		target = torch.from_numpy(YY)
		dataset = TensorDataset(torch.unsqueeze(data,1), target)
	elif model_type == 'RNN':
		data = torch.from_numpy(XX.astype(np.float32))
		target = torch.from_numpy(YY)
		dataset = TensorDataset(torch.unsqueeze(data,2), target)

	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	num_features = reduce(lambda m,n: m + n, seqs)
	return len(set(np.concatenate(num_features)))


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels
		# Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# can use Sparse matrix type for memory efficiency if you want.
		# self.seqs = [i for i in range(len(labels))]  # replace this with your implementation.

		n = 0
		all_sparse = []
		for p in seqs:
			sparse_data = np.zeros((len(p), num_features))
			for a, b in enumerate(p):
				sparse_data[a, b] = 1
			all_sparse.append(sparse_data)
			n += 1
		self.seqs = all_sparse


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# Return the following two things
	# 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# 2. Tensor contains the label of each sequence

	sizes = []
	maxLengths = []
	features = []
	sample_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
	for i, j in sample_batch:
		sizes.append(j)
		maxLengths.append(i.shape[0])
		features.append(i.shape[1])
	batch_size = len(sizes)
	batch_maxLength = np.max(maxLengths)
	num_features = np.max(features)
	Matrix_3D = np.zeros((batch_size, batch_maxLength, num_features))

	Ind = 0
	lengths = []
	labels = []
	for Unit, Label in sample_batch:
		lengths.append(len(Unit))
		labels.append(Label)
		Matrix_3D[Ind, 0:len(Unit), :] = Unit
		Ind += 1

	seqs_tensor = torch.FloatTensor(Matrix_3D)
	lengths_tensor = torch.LongTensor(lengths)
	labels_tensor = torch.LongTensor(labels)

	return (seqs_tensor, lengths_tensor), labels_tensor
