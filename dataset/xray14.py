import torch
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Xray14Dataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the Chest X-ray 14 dataset
	"""

	def __init__(
		self,
		pkl_file_path,
		dataset_root,
		transform=None,
		return_orig=False,
		resize_orig=True,
		return_path=False,
		return_points=False,
		bb_only=False):
		"""
		Args:
			pkl_file_path (string): path to pkl file for dataset split from repo root
			dataset_root (string): path to dataset root from repo root
			transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
			return_orig (bool): If True then return the original image
			resize_orig (bool): If True then resize the returned original image to the same size as the transformed image
			return_path (bool): If True then return the original image path (not used if return_orig is true)
			return_points (bool): If True then return playing card location points (not used if return_orig or return_path is true)
			bb_only (bool): If true then only return samples with bounding boxes
		"""
		self.data = []
		self.transform = transform
		self.return_orig = return_orig
		self.resize_orig = resize_orig
		self.return_path = return_path
		self.return_points = return_points
		self.is_training = True if ('train.pkl' in pkl_file_path) else False

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		highest_class = 0
		for i in dataset_split:
			sample = {
				"id": dataset_split[i],
				"img_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["img_path"],
				"class_index": dataset_split[i]["class_label"],  # class are 0 indexed
				"concepts": dataset_split[i]["concept_label"],
				#"bounding_box": dataset_split[i]["bounding_box"],
			}

			if dataset_split[i]["class_label"] > highest_class:
				highest_class = dataset_split[i]["class_label"]

			# Only keep samples with bounding boxes if bb_only is set to True
			if (bb_only and (sample["bounding_box"] != None)):
				self.data.append(sample)
			elif bb_only is False:
				self.data.append(sample)

		self.num_concepts = len(self.data[0]["concepts"])
		self.num_classes = highest_class + 1

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_path = img_data['img_path']

		#img = Image.open(img_path).convert('L')
		img = Image.open(img_path).convert('RGB')
		class_index = img_data['class_index']
		concepts = img_data['concepts']

		x_resize = 512
		y_resize = 512

		# create lists for sign points and classes
		# points are adjusted for resized images
		"""
		if self.return_points:
			signs_points = []
			signs_classes = []
			for sign in img_data['bounding_box']:
				points = []
				for point in sign[0:-1]:  # last item in list is sign concept id
					point = (int(point[0] * (int(x_resize)/img.width)), int(point[1] * (int(y_resize)/img.height)))
					points.append(point)
				signs_points.append(points)
				signs_classes.append(sign[4])

			# The dataloader needs every sample to have points and classes for all concepts.
			# pad / create tensors to include values for all concepts.
			# The index for points and classes will match. If a class has a value of -1 then the points are padding (not from in the dataset.)
			if len(signs_points) > 0:

				signs_points = torch.as_tensor(signs_points)
				signs_classes = torch.as_tensor(signs_classes)

				points_target_size = (self.num_concepts, 4, 2)  # num conceps, num points, x and y

				signs_points = F.pad(signs_points, (0, points_target_size[2] - signs_points.size(2), 0, points_target_size[1] - signs_points.size(1), 0, points_target_size[0] - signs_points.size(0)))
				signs_classes = F.pad(signs_classes, (0, self.num_concepts - signs_classes.size(0)), value=-1)
			else:
				signs_points = torch.zeros(self.num_concepts, 4, 2)
				signs_classes = torch.ones(self.num_concepts) * -1
		"""

		if self.transform:
			altered_img = self.transform(img)
		else:
			altered_img = img

		if self.return_orig or self.return_path or self.return_points:
			extra = {}
			if self.return_orig:
				img = transforms.ToTensor()(img.convert('RGB'))
				if self.resize_orig:
					extra["orig_img"] = torch.as_tensor(transforms.Resize((x_resize, y_resize))(img))
				else:
					extra["orig_img"] = torch.as_tensor(img)
			if self.return_path:
				extra["path"] = img_path
			#if self.return_points:
			#	extra["points"] = signs_points
			#	extra["sign_classes"] = signs_classes

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
