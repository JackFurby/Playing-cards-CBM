import os

import numpy as np
import cv2


class IndexToString(object):
	"""Convert index to the corrisponding string
	Args:
		file_path (string): path to txt file containing class ids (0 indexed) and strings
	"""

	def __init__(self, file_path):
		self.string_dict = {}
		with open(file_path) as f:
			lines = f.readlines()
			for i in lines:
				i = i.split(" ")
				self.string_dict[int(i[1])] = i[0]

	def __call__(self, index):
		"""
		Args:
			index (int): index of class
		Returns:
			String: String name
		"""
		return self.string_dict[index]


class StringToIndex(object):
	"""Convert string to the corrisponding index
	Args:
		file_path (string): path to txt file containing class ids (0 indexed) and strings
	"""

	def __init__(self, file_path):
		self.string_dict = {}
		with open(file_path) as f:
			lines = f.readlines()
			for i in lines:
				i = i.split(" ")
				self.string_dict[i[0]] = int(i[1])

	def __call__(self, string):
		"""
		Args:
			string (str): string of index
		Returns:
			int: index
		"""
		return self.string_dict[string]




# https://github.com/jfhealthcare/Chexpert/blob/master/data/utils.py
def border_pad(image, long_side, border_pad_str, pixel_mean):
	h, w, c = image.shape

	if border_pad == 'zero':
		image = np.pad(image, ((0, long_side - h), (0, long_side - w), (0, 0)), mode='constant', constant_values=0.0)
	elif border_pad == 'pixel_mean':
		image = np.pad(image, ((0, long_side - h), (0, long_side - w), (0, 0)), mode='constant', constant_values=pixel_mean)
	else:
		image = np.pad(image, ((0, long_side - h), (0, long_side - w), (0, 0)), mode=border_pad_str)

	return image


# https://github.com/jfhealthcare/Chexpert/blob/master/data/utils.py
def fix_ratio(image, long_side, border_pad_str, pixel_mean):
	h, w, c = image.shape

	if h >= w:
		ratio = h * 1.0 / w
		h_ = long_side
		w_ = round(h_ / ratio)
	else:
		ratio = w * 1.0 / h
		w_ = long_side
		h_ = round(w_ / ratio)

	image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
	image = border_pad(image, long_side, border_pad_str, pixel_mean)

	return image


# https://github.com/jfhealthcare/Chexpert/blob/master/data/utils.py
def transform(image, gaussian_blur, use_equalizeHist, long_side, border_pad_str, pixel_mean, pixel_std=None):
	assert image.ndim == 2, "image must be gray image"
	if use_equalizeHist:
		image = cv2.equalizeHist(image)

	if gaussian_blur > 0:
		image = cv2.GaussianBlur(image, (gaussian_blur, gaussian_blur), 0)

	image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

	image = fix_ratio(image, long_side, border_pad_str, pixel_mean)
	# augmentation for train or co_train

	# normalization
	image = image.astype(np.float32) - pixel_mean
	# vgg and resnet do not use pixel_std, densenet and inception use.
	if pixel_std:
		image /= pixel_std
	# normal image tensor :  H x W x C
	# torch image tensor :   C X H X W
	image = image.transpose((2, 0, 1))

	return image
