import torch
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image


class ChexpertDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the CelebA dataset
	"""

	def __init__(
		self,
		pkl_file_path,
		dataset_root,
		transform=None,
		keep_concepts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
		return_orig=False,
		resize_orig=True,
		return_path=False):
		"""
		Args:
			pkl_file_path (string): path to pkl file for dataset split from repo root
			dataset_root (string): path to dataset root from repo root
			transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
			keep_concepts (list of ints): list of all indexes (0 indexed) of concepts to use. There is a maximum of 13 conecpts.
			return_orig (bool): If True then return the original image
			resize_orig (bool): If True then resize the returned original image to the same size as the transformed image
			return_path (bool): If True then return the original image path (not used if return_orig is true)
		"""
		self.data = []
		self.transform = transform
		self.return_orig = return_orig
		self.resize_orig = resize_orig
		self.return_path = return_path
		self.keep_concepts = keep_concepts

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		highest_class = 0
		for i in dataset_split:

			# discard any concepts we are not keeping
			final_concepts = []
			for idx, concept in enumerate(dataset_split[i]["concept_label"]):
				if idx in self.keep_concepts:
					final_concepts.append(concept)

			sample = {
				"id": dataset_split[i],
				"img_path": os.getcwd() + "/" + dataset_root + "/" + dataset_split[i]["img_path"],
				"class_index": dataset_split[i]["class_label"],  # class are 0 indexed
				"concepts": final_concepts,
			}
			self.data.append(sample)

			if dataset_split[i]["class_label"] > highest_class:
				highest_class = dataset_split[i]["class_label"]

		self.num_classes = highest_class + 1

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_path = img_data['img_path']

		#img = Image.open(img_path).convert('RGB')
		img = Image.open(img_path)
		class_index = img_data['class_index']
		concepts = img_data['concepts']

		if self.transform:
			#altered_img = self.transform(img)
			altered_img = self.transform(np.asarray(img), gaussian_blur=3, use_equalizeHist=True, long_side=512, border_pad_str="mean", pixel_mean=128.0, pixel_std=64.0)
		else:
			altered_img = img

		if self.return_orig or self.return_path:
			extra = {}
			if self.return_orig:
				#img = transforms.ToTensor()(img)
				img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
				if self.resize_orig:
					extra["orig_img"] = torch.as_tensor(transforms.Resize((218, 178))(img))
				else:
					extra["orig_img"] = torch.as_tensor(img)
			if self.return_path:
				extra["path"] = img_paths

			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), extra
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)
