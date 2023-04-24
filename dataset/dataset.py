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
		return_orig=False,
		resize_orig=True,
		return_path=False,
		return_points=False):
		"""
		Args:
			pkl_file_path (string): path to pkl file for dataset split from repo root
			dataset_root (string): path to dataset root from repo root
			transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
			return_orig (bool): If True then return the original image
			resize_orig (bool): If True then resize the returned original image to the same size as the transformed image
			return_path (bool): If True then return the original image path (not used if return_orig is true)
			return_points (bool): If True then return playing card location points (not used if return_orig or return_path is true)
		"""
		self.data = []
		self.transform = transform
		self.return_orig = return_orig
		self.resize_orig = resize_orig
		self.return_path = return_path
		self.return_points = return_points

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		highest_class = 0
		for i in dataset_split:
			sample = {
				"id": dataset_split[i],
				"img_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["img_path"],
				"class_index": dataset_split[i]["class_label"],  # class are 0 indexed
				"concepts": dataset_split[i]["concept_label"],
				"card_points": dataset_split[i]["card_points"],
			}
			self.data.append(sample)

			if dataset_split[i]["class_label"] > highest_class:
				highest_class = dataset_split[i]["class_label"]

		self.num_concepts = len(self.data[0]["concepts"])
		self.num_classes = highest_class + 1

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_path = img_data['img_path']

		img = Image.open(img_path).convert('RGB')
		class_index = img_data['class_index']
		concepts = img_data['concepts']

		# temp - convert concepts to 3 hot vector
		#concepts = torch.zeros(52)
		#for i in img_data['concepts']:
		#	concepts[i] = 1

		# create lists for card points and card classes
		# card points are adjusted for resized images
		if self.return_points:
			card_points = []
			card_classes = []
			for card in img_data['card_points']:
				points = []
				for point in card[0]:
					point = (int(point[0] * (int(299)/img.width)), int(point[1] * (int(299)/img.height)))
					points.append(point)
				card_points.append(points)
				card_classes.append(card[1])

		if self.transform:
			altered_img = self.transform(img)
		else:
			altered_img = img

		if self.return_orig or self.return_path or self.return_points:
			extra = {}
			if self.return_orig:
				img = transforms.ToTensor()(img)
				if self.resize_orig:
					extra["orig_img"] = torch.as_tensor(transforms.Resize((299, 299))(img))
				else:
					extra["orig_img"] = torch.as_tensor(img)
			if self.return_path:
				extra["path"] = img_path
			if self.return_points:
				extra["points"] = torch.as_tensor(card_points, dtype=torch.float)
				extra["card_classes"] = torch.as_tensor(card_classes)

			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), extra
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)


class CelebADataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the CelebA dataset
	"""

	def __init__(
		self,
		pkl_file_path,
		dataset_root,
		transform=None,
		return_orig=False,
		resize_orig=True,
		return_path=False):
		"""
		Args:
			pkl_file_path (string): path to pkl file for dataset split from repo root
			dataset_root (string): path to dataset root from repo root
			transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
			return_orig (bool): If True then return the original image
			resize_orig (bool): If True then resize the returned original image to the same size as the transformed image
			return_path (bool): If True then return the original image path (not used if return_orig is true)
		"""
		self.data = []
		self.transform = transform
		self.return_orig = return_orig
		self.resize_orig = resize_orig
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
		concepts = img_data['concepts']

		if self.transform:
			altered_img = self.transform(img)
		else:
			altered_img = img

		if self.return_orig or self.return_path:
			extra = {}
			if self.return_orig:
				img = transforms.ToTensor()(img)
				if self.resize_orig:
					extra["orig_img"] = torch.as_tensor(transforms.Resize((218, 178))(img))
				else:
					extra["orig_img"] = torch.as_tensor(img)
			if self.return_path:
				extra["path"] = img_paths

			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), extra
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Args:
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
def load_data(pkl_path, batch_size, dataset_root='dataset/playing_cards', dataset_name="cards", x_resize=299, y_resize=299, return_orig=False, resize_orig=True, return_path=False, return_points=False, resampling=False, shuffle=None, num_workers=0, drop_last=None, train_transform=True):
	"""
	"""
	is_training = True if ('train.pkl' in pkl_path and train_transform) else False
	if dataset_name=="cards":
		if is_training:
			transform = transforms.Compose([
				transforms.Resize((y_resize, x_resize)),
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(),
				transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.3),
				transforms.RandomGrayscale(p=0.1),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
			])
		else:
			transform = transforms.Compose([
				transforms.Resize((y_resize, x_resize)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
			])
	else:
		if is_training:
			transform = transforms.Compose([
				transforms.Resize((y_resize, x_resize)),
				#transforms.RandomResizedCrop((y_resize, x_resize)),
				#transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
				#transforms.RandomHorizontalFlip(),
				transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.09, 1.05))
				transforms.ToTensor(), #implicitly divides by 255
				transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])
		else:
			transform = transforms.Compose([
				transforms.Resize((y_resize, x_resize)),
				#transforms.CenterCrop((y_resize, x_resize)),
				transforms.ToTensor(), #implicitly divides by 255
				transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])

	if dataset_name == "cards":
		dataset = PlayingCardDataset(pkl_path, dataset_root, transform, return_orig=return_orig, resize_orig=resize_orig, return_path=return_path, return_points=return_points)
	else:
		dataset = CelebADataset(pkl_path, dataset_root, transform, return_orig=return_orig, resize_orig=resize_orig, return_path=return_path)
	if is_training:
		drop_last = True if drop_last is None else drop_last
		shuffle = True if shuffle is None else shuffle
	else:
		drop_last = False if drop_last is None else drop_last
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
