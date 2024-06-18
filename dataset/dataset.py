import torch
import torchvision.transforms as transforms

from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
import pickle

from dataset.utils import transform as chexpert_transform
from dataset.chexpert import ChexpertDataset
from dataset.cub import CubDataset
from dataset.playing_cards import PlayingCardDataset
from dataset.skin import SkinDataset
from datast.blackjack import BlackjackDataset


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
def load_data(pkl_path, batch_size, dataset_root='dataset/playing_cards', dataset_name="cards", x_resize=299, y_resize=299, return_orig=False, resize_orig=True, return_path=False, return_points=False, resampling=False, shuffle=None, num_workers=0, drop_last=None, train_transform=True, keep_concepts=[], bb_only=False):
	"""
	"""
	is_training = True if ('train.pkl' in pkl_path and train_transform) else False
	if dataset_name in ["cards", "blackjack"]:
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
	elif dataset_name=="CUB":
		if is_training:
			transform = transforms.Compose([
				transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
				transforms.Resize(x_resize + 25),
				transforms.CenterCrop(x_resize),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(), #implicitly divides by 255
				transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
				])
		else:
			transform = transforms.Compose([
				transforms.Resize(x_resize + 25),
				transforms.CenterCrop(x_resize),
				transforms.ToTensor(), #implicitly divides by 255
				transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
				])
	elif dataset_name == "chexpert":
		if is_training:
			transform = chexpert_transform
		else:
			transform = chexpert_transform
	elif dataset_name == "skin":
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		if is_training:
			transform = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(brightness=0.1),
				transforms.Resize(x_resize),
				transforms.CenterCrop(x_resize),
				transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
				transforms.ToTensor(),
				transforms.Normalize(mean, std)
			])
		else:
			transform = transforms.Compose([
				transforms.Resize(x_resize),
				transforms.CenterCrop(x_resize),
				transforms.ToTensor(),
				transforms.Normalize(mean, std)
			])
	else:
		if is_training:
			transform = transforms.Compose([
				transforms.Resize(x_resize + 25),
				transforms.CenterCrop(x_resize),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(brightness=0.1),
				transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
				transforms.ToTensor(), #implicitly divides by 255
				transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])
		else:
			transform = transforms.Compose([
				transforms.Resize(x_resize + 25),
				transforms.CenterCrop(x_resize),
				transforms.ToTensor(), #implicitly divides by 255
				transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			])

	if dataset_name == "cards":
		dataset = PlayingCardDataset(pkl_path, dataset_root, transform, return_orig=return_orig, resize_orig=resize_orig, return_path=return_path, return_points=return_points)
	if dataset_name == "blackjack":
		dataset = BlackjackDataset(pkl_path, dataset_root, transform, return_orig=return_orig, resize_orig=resize_orig, return_path=return_path, return_points=return_points)
	elif dataset_name == "CUB":
		dataset = CubDataset(pkl_path, dataset_root, transform, return_orig=return_orig, resize_orig=resize_orig, return_path=return_path)
	elif dataset_name == "chexpert":
		dataset = ChexpertDataset(pkl_path, dataset_root, transform, return_orig=return_orig, return_path=return_path)
	elif dataset_name == "skin":
		if len(keep_concepts) == 0:
			dataset = SkinDataset(pkl_path, dataset_root, transform=transform, return_orig=return_orig, resize_orig=resize_orig, return_path=return_path, return_points=return_points)
		else:
			dataset = SkinDataset(pkl_path, dataset_root, transform=transform, return_orig=return_orig, resize_orig=resize_orig, return_path=return_path, return_points=return_points)
	else:
		raise Exception(f"Unknown dataset type: {dataset_name}")
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
