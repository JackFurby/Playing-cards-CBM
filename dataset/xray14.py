import torch
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


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
		self.is_training = True if ('train.pkl' in pkl_file_path) else False

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		highest_class = 0
		for i in dataset_split:
			sample = {
				"id": dataset_split[i],
				"img_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["img_path"],
				"class_index": dataset_split[i]["class_label"],  # class are 0 indexed
				"concepts": dataset_split[i]["concept_label"],
				"bounding_box": dataset_split[i]["bounding_box"],
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

		img = Image.open(img_path)
		class_index = img_data['class_index']
		concepts = img_data['concepts']

		# create lists for sign points and classes
		# points are adjusted for resized images
		if self.return_points:
			sign_points = []
			sign_classes = []
			for sign in img_data['bounding_box']:
				points = []
				for point in sign[0]:
					point = (int(point[0] * (int(512)/img.width)), int(point[1] * (int(512)/img.height)))
					points.append(point)
				sign_points.append(points)
				sign_classes.append(sign[1])

		if self.transform:
			if self.is_training:
				img = transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)(img)
			altered_img = self.transform(np.asarray(img), gaussian_blur=3, use_equalizeHist=True, long_side=512, border_pad_str="mean", pixel_mean=128.0, pixel_std=64.0)
		else:
			altered_img = img

		if self.return_orig or self.return_path or self.return_points:
			extra = {}
			if self.return_orig:
				img = transforms.ToTensor()(img)
				if self.resize_orig:
					extra["orig_img"] = torch.as_tensor(transforms.Resize((512, 512))(img))
				else:
					extra["orig_img"] = torch.as_tensor(img)
			if self.return_path:
				extra["path"] = img_path
			if self.return_points:
				extra["points"] = torch.as_tensor(card_points, dtype=torch.float)
				extra["sign_classes"] = torch.as_tensor(card_classes)

			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), extra
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)
