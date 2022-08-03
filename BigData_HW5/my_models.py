import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.layer1 = nn.Linear(178, 32)  # Improved model
		self.out = nn.Linear(32, 5)
		self.dropout1 = nn.Dropout(p = 0.2)
		self.dropout2 = nn.Dropout(p = 0.1)

	def forward(self, x):
		x = F.relu(self.dropout1(self.layer1(x)))    # Improved model
		x = self.dropout2(x)
		x = self.out(x)
		return x

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()

		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)     # Improved model
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
		self.fc1 = nn.Linear(in_features=16*41, out_features=128)
		self.fc2 = nn.Linear(128, 5)
		self.dropout1 = nn.Dropout(p=0.2)
		self.dropout2 = nn.Dropout(p=0.5)

	def forward(self, x):
		x = self.pool(self.dropout1(F.relu(self.conv1(x))))                              # Improved model
		x = self.pool(self.dropout1(F.relu(self.conv2(x))))
		x = x.view(-1, 16*41)
		x = self.dropout2(F.relu(self.fc1(x)))
		x = self.fc2(x)
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.Rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)           # Improved model
		self.Fc = nn.Linear(in_features=16, out_features=5)
		self.dropout1 = nn.Dropout(p=0.3)

	def forward(self, x):
		x, _ = self.Rnn(x)
		x = self.dropout1(F.relu(x[:, -1, :]))
		x = self.Fc(x)
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.dim_input = dim_input                                                      # Improved model
		self.FC1 = nn.Linear(in_features=dim_input, out_features=32)
		self.Rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
		self.FC2 = nn.Linear(in_features=16, out_features=2)
		self.dropout1 = nn.Dropout(p=0.7)
		self.dropout2 = nn.Dropout(p=0.1)
		self.dropout3 = nn.Dropout(p=0.5)
		self.dropout4 = nn.Dropout(p=0.7)


	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		seqs = self.dropout1(F.relu(seqs))
		seqs = torch.tanh(self.FC1(seqs))
		seqs = self.dropout2(seqs)
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
		s, h = self.Rnn(seqs)

		output, input = pad_packed_sequence(s, batch_first=True)
		idxs = torch.LongTensor([x - 1 for x in input])
		result = output[range(output.shape[0]), idxs, :]
		result = self.dropout4(result)
		result = self.dropout3(F.relu(result))
		result = self.FC2(result)
		return result



