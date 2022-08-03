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
				"class_index": dataset_split[i]["class_label"],  # class are 0 indexed
				"concepts": dataset_split[i]["concept_label"],
			}
			self.data.append(sample)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_path = img_data['img_path']

		img = Image.open(img_path).convert('RGB')
		class_index = img_data['class_index']
		#concepts = img_data['concepts']

		# temp - convert concepts to 3 hot vector
		concepts = torch.zeros(52)
		for i in img_data['concepts']:
			concepts[i] = 1


		if self.transform is not None:
			altered_img = self.transform(img)
		else:
			altered_img = img

		if self.return_orig:
			# hard coded transform without normalise (can I make this better?)
			img = transforms.ToTensor()(img)
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), torch.as_tensor(transforms.Resize((299, 299))(img))
		elif self.return_path:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), img_path
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)


class MultiImagePlayingCardDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the playing card dataset where three images are provided per sample
	"""

	def __init__(
		self,
		pkl_file_path,
		dataset_root,
		num_concepts,
		transform=None,
		train=True,
		return_orig=False):
		"""
		Args:
			pkl_file_path (string): path to pkl file for dataset split from repo root
			dataset_root (string): path to dataset root from repo root
			num_concepts (int): number of concepts per image
			transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
			train (bool): True if dataset contains training data else False
			return_orig (bool): If True then return the original image
		"""
		self.data = []
		self.train = train
		self.transform = transform
		self.return_orig = return_orig
		self.num_concepts = num_concepts

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		for i in dataset_split:
			sample = {
				"id": dataset_split[i],
				"img_1_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["card_1_img_path"],
				"img_2_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["card_2_img_path"],
				"img_3_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["card_3_img_path"],
				"card_1_class_index": dataset_split[i]["card_1_class_label"],  # class are 0 indexed
				"card_2_class_index": dataset_split[i]["card_2_class_label"],  # class are 0 indexed
				"card_3_class_index": dataset_split[i]["card_3_class_label"],  # class are 0 indexed
				"card_1_concepts": dataset_split[i]["card_1_concepts"],
				"card_2_concepts": dataset_split[i]["card_2_concepts"],
				"card_3_concepts": dataset_split[i]["card_3_concepts"],
				"class_label": dataset_split[i]["class_label"]  # hand ranks are 0 indexed

			}
			self.data.append(sample)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_1_path = img_data['img_1_path']
		img_2_path = img_data['img_2_path']
		img_3_path = img_data['img_3_path']

		img_1 = Image.open(img_1_path).convert('RGB')
		img_2 = Image.open(img_2_path).convert('RGB')
		img_3 = Image.open(img_3_path).convert('RGB')
		class_label = img_data['class_label']
		card_1_index = img_data['card_1_class_index']
		card_2_index = img_data['card_2_class_index']
		card_3_index = img_data['card_3_class_index']

		# convert card classes to one hot vectors
		card_1_concept = torch.zeros(self.num_concepts)
		card_1_concept[img_data['card_1_class_index']] += 1
		card_2_concept = torch.zeros(self.num_concepts)
		card_2_concept[img_data['card_2_class_index']] += 1
		card_3_concept = torch.zeros(self.num_concepts)
		card_3_concept[img_data['card_3_class_index']] += 1

		card_concepts = torch.stack([card_1_concept, card_2_concept, card_3_concept])

		if self.transform is not None:
			altered_img_1 = self.transform(img_1)
			altered_img_2 = self.transform(img_2)
			altered_img_3 = self.transform(img_3)
		else:
			altered_img_1 = img_1
			altered_img_2 = img_2
			altered_img_3 = img_3

		if self.return_orig:
			# hard coded transform without normalise (can I make this better?)
			img_1 = transforms.ToTensor()(img_1)
			img_2 = transforms.ToTensor()(img_2)
			img_3 = transforms.ToTensor()(img_3)
			return torch.as_tensor(torch.stack((altered_img_1, altered_img_2, altered_img_3),dim=0)), torch.as_tensor(class_label), torch.as_tensor(card_concepts), torch.as_tensor(torch.stack((img_1, img_2, img_3),dim=0))
		else:
			return torch.as_tensor(torch.stack((altered_img_1, altered_img_2, altered_img_3),dim=0)), torch.as_tensor(class_label), torch.as_tensor(card_concepts)


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
		return dataset.data[idx]["id"]['concept_label'][0]

	def __iter__(self):
		idx = (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))
		return idx

	def __len__(self):
		return self.num_samples


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
def load_data(pkl_path, batch_size, num_concepts, dataset_root='dataset/playing_cards', return_orig=False, return_path=False, resampling=False, shuffle=None, num_workers=0, multi_image=False):
	"""
	"""
	is_training = True if 'train.pkl' in pkl_path else False
	if is_training:
		transform = transforms.Compose([
			transforms.Resize((299, 299)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])
	else:
		transform = transforms.Compose([
			transforms.Resize((299, 299)),
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])

	if multi_image:
		dataset = MultiImagePlayingCardDataset(pkl_path, dataset_root, num_concepts, transform, train=is_training, return_orig=return_orig)
	else:
		dataset = PlayingCardDataset(pkl_path, dataset_root, transform, train=is_training, return_orig=return_orig, return_path=return_path)
	if is_training:
		drop_last = True
		shuffle = True if shuffle is None else shuffle
	else:
		drop_last = False
		shuffle = False if shuffle is None else shuffle
	if resampling:
		sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
		loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
	else:
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
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
	n_attr = len(list(data.items())[0][1]['concept_label'])
	if attr_idx >= 0:
		n_attr = 1
	if multiple_attr:
		n_ones = [0] * n_attr
		total = [n] * n_attr
	else:
		n_ones = [0]
		total = [n * n_attr]
	for key in data:
		labels = data[key]['concept_label']
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
