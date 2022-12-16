import torch
from dataset.dataset import load_data
from dataset.utils import *
import matplotlib.pyplot as plt
import numpy as np


def get_device(cpu=False):
	device = torch.device("cuda:0" if (torch.cuda.is_available() and not cpu) else "cpu")
	if torch.cuda.is_available():
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)
	return device


def get_dataset(split, return_points=False, return_orig=False, return_path=False, shuffle=True):
	if split == "single":
		n_concepts = 17
		n_classes = 52
		val_loader = load_data("../dataset/playing-cards/splits/single/val.pkl", 16, dataset_root="../dataset/playing-cards/", return_points=return_points, return_orig=return_orig, return_path=return_path, shuffle=shuffle)
		concept_index_to_string = IndexToString("../dataset/playing-cards/splits/single/concepts.txt")
	elif split == "three":
		n_concepts = 52
		n_classes = 1
		val_loader = load_data("../dataset/playing-cards/splits/three/val.pkl", 16, dataset_root="../dataset/playing-cards/", has_class=False, return_points=return_points, return_orig=return_orig, return_path=return_path, shuffle=shuffle)
		concept_index_to_string = IndexToString("../dataset/playing-cards/splits/three/concepts.txt")
	elif split == "poker":
		n_concepts = 52
		n_classes = 6
		val_loader = load_data("../dataset/playing-cards/splits/three_card_poker/val.pkl", 16, dataset_root="../dataset/playing-cards/", return_points=return_points, return_orig=return_orig, return_path=return_path, shuffle=shuffle)
		concept_index_to_string = IndexToString("../dataset/playing-cards/splits/three_card_poker/concepts.txt")
	
	return n_concepts, n_classes, val_loader, concept_index_to_string

##### https://github.com/fhvilshoj/TorchLRP/blob/74253a1be05f0be0b7c535736023408670443b6e/examples/visualization.py#L60
def heatmap(X, cmap_name="seismic"):
	cmap = plt.cm.get_cmap(cmap_name)

	if X.shape[1] in [1, 3]:  # move channel index to end + convert to np array
		X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
	if isinstance(X, torch.Tensor):  # convert tensor to np array
		X = X.detach().cpu().numpy()

	shape = X.shape
	tmp = X.sum(axis=-1) # Reduce channel axis

	tmp = project(tmp, output_range=(0, 255)).astype(int)
	tmp = cmap(tmp.flatten())[:, :3].T
	tmp = tmp.T

	shape = list(shape)
	shape[-1] = 3
	return tmp.reshape(shape).astype(np.float32)


def project(X, output_range=(0, 1)):
	absmax = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
	X /= absmax + (absmax == 0).astype(float)
	X = (X+1) / 2. # range [0, 1]
	X = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
	return X