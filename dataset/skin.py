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


class SkinDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the SkinCon dataset
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
		self.is_training = True if ('train.pkl' in pkl_file_path) else False

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		highest_class = 0
		for i in dataset_split:
			sample = {
				"id": dataset_split[i],
				"img_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["img_path"],
				"class_index": dataset_split[i]["class_label"],  # class are 0 indexed
				"concepts": dataset_split[i]["concept_label"],
			}

			if dataset_split[i]["class_label"] > highest_class:
				highest_class = dataset_split[i]["class_label"]

			self.data.append(sample)

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

		x_resize = 512
		y_resize = 512

		if self.transform:
			altered_img = self.transform(img)
		else:
			altered_img = img

		if self.return_orig or self.return_path:
			extra = {}
			if self.return_orig:
				img = transforms.ToTensor()(img.convert('RGB'))
				if self.resize_orig:
					extra["orig_img"] = torch.as_tensor(transforms.Resize((x_resize, y_resize))(img))
				else:
					extra["orig_img"] = torch.as_tensor(img)
			if self.return_path:
				extra["path"] = img_path

			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), extra
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)
