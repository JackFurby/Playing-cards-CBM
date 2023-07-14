import torch
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image


class CubDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the CUB dataset
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
					orig_transform = transforms.Compose([
						transforms.Resize(299 + 25),
						transforms.CenterCrop(299)
					])
					extra["orig_img"] = torch.as_tensor(orig_transform(img))
				else:
					extra["orig_img"] = torch.as_tensor(img)
			if self.return_path:
				extra["path"] = img_paths

			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), extra
		else:
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)
