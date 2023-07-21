import torch
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

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
		keep_concepts=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
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
			keep_concepts list(ints): Only keep concepts in this list (0 indexed). Concepts will be returned in the order provided
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
				"bounding_box": dataset_split[i]["bounding_box"],
			}

			new_concepts = []
			if sample["bounding_box"] == None:
				new_bb = None
			else:
				new_bb = []
			for new_index, j in enumerate(keep_concepts):
				new_concepts.append(sample["concepts"][j])  # Add concepts to concept to keep
				# only keep bounding boxes in keep_concepts
				if new_bb != None:
					bb_index = [k[4] for k in sample["bounding_box"]]
					for idx, concept in enumerate(bb_index):
						if j == concept:
							new_bb_value = sample["bounding_box"][idx][0:-1]
							new_bb_value.append(new_index)
							new_bb.append(new_bb_value)

			## bb is for a discarded concept
			if (new_bb != None) and (len(new_bb) == 0):
				new_bb = None

			sample["concepts"] = new_concepts
			sample["bounding_box"] = new_bb

			# Only keep samples with bounding boxes if bb_only is set to True
			if (bb_only and (sample["bounding_box"] != None)):
				self.data.append(sample)
			elif bb_only is False:
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

		img = Image.open(img_path).convert('L')
		class_index = img_data['class_index']
		concepts = img_data['concepts']

		x_resize = 224
		y_resize = 224

		# create lists for sign points and classes
		# points are adjusted for resized images
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

		if self.transform:
			altered_img = img
			if self.is_training:
				altered_img = transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)(altered_img)
			altered_img = self.transform(np.asarray(altered_img), gaussian_blur=3, use_equalizeHist=True, long_side=512, border_pad_str="mean", pixel_mean=128.0, pixel_std=64.0)
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
			if self.return_points:
				extra["points"] = signs_points
				extra["sign_classes"] = signs_classes

			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), extra
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)
