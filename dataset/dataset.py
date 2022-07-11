import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import pickle


class PlayingCardDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the playing card dataset
	"""

	def __init__(
		self,
		pkl_file_path,
		dataset_root,
		transform=None,
		has_class=True,
		train=True,
		return_orig=False,
		return_path=False):
		"""
		Args:
			pkl_file_path (string): path to pkl file for dataset split from repo root
			dataset_root (string): path to dataset root from repo root
			transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
			train (bool): True if dataset contains training data else False
			return_orig (bool): If True then return the original image
			return_path (bool): If True then return the original image path (not used if return_orig is true)
		"""
		self.data = []
		self.train = train
		self.transform = transform
		self.return_orig = return_orig
		self.return_path = return_path

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		for i in dataset_split:
			sample = {
				"id": dataset_split[i],
				"img_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["img_path"],
				"class_index": dataset_split[i]["class_label"] if has_class else 0,  # class are 0 indexed
				"concepts": dataset_split[i]["attribute_label"],
			}
			self.data.append(sample)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_path = img_data['img_path']

		img = Image.open(img_path).convert('RGB')
		class_index = img_data['class_index']
		concepts = img_data['concepts']

		if self.transform is not None:
			altered_img = self.transform(img)
		else:
			altered_img = img

		if self.return_orig:
			# hard coded transform without normalise (can I make this better?)
			img = transforms.ToTensor()(img)
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), torch.as_tensor(img)
		elif self.return_path:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), img_path
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None):
		# if indices is not provided,
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided,
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)

		# distribution of classes in the dataset
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1

		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
			for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):  # Note: for single attribute dataset
		return dataset.data[idx]["id"]['attribute_label'][0]

	def __iter__(self):
		idx = (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))
		return idx

	def __len__(self):
		return self.num_samples


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
def load_data(pkl_path, batch_size, dataset_root='dataset/playing_cards', return_orig=False, return_path=False, has_class=True, resampling=False):
	"""
	"""
	is_training = True if 'train.pkl' in pkl_path else False
	if is_training:
		transform = transforms.Compose([
			#transforms.Resize((274, 174)),
			transforms.Resize((280, 280)),
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])
	else:
		transform = transforms.Compose([
			#transforms.Resize((274, 174)),
			transforms.Resize((280, 280)),
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])

	dataset = PlayingCardDataset(pkl_path, dataset_root, transform, has_class=has_class, train=is_training, return_orig=return_orig, return_path=return_path)
	if is_training:
		drop_last = True
		shuffle = True
	else:
		drop_last = False
		shuffle = False
	if resampling:
		sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
		loader = DataLoader(dataset, batch_sampler=sampler)
	else:
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
	return loader


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
	"""
	Calculate class imbalance ratio for binary attribute labels stored in pkl_file
	If attr_idx >= 0, then only return ratio for the corresponding attribute id
	If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
	"""
	imbalance_ratio = []
	data = pickle.load(open(pkl_file, 'rb'))
	n = len(data)
	n_attr = len(list(data.items())[0][1]['attribute_label'])
	if attr_idx >= 0:
		n_attr = 1
	if multiple_attr:
		n_ones = [0] * n_attr
		total = [n] * n_attr
	else:
		n_ones = [0]
		total = [n * n_attr]
	for key in data:
		labels = data[key]['attribute_label']
		if multiple_attr:
			for i in range(n_attr):
				n_ones[i] += labels[i]
		else:
			if attr_idx >= 0:
				n_ones[0] += labels[attr_idx]
			else:
				n_ones[0] += sum(labels)
	for j in range(len(n_ones)):
		imbalance_ratio.append(total[j]/n_ones[j] - 1)
	if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
		imbalance_ratio *= n_attr
	return imbalance_ratio
