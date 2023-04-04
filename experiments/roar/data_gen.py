import torch
import argparse
from tqdm import tqdm
import itertools
from matplotlib import pyplot as plt
import random
import os
from dataset.dataset import load_data
from torchvision.utils import save_image
#from models.model import End2EndModel
from models.vgg import End2EndModel
from models.lrp_converter import convert
import lrp
from experiments.utils import heatmap
import cv2
from captum.attr import IntegratedGradients, NoiseTunnel
from distutils.dir_util import copy_tree

import time
import numpy as np

from matplotlib.patches import Circle


def replace_fetures(img, coords, device="cpu"):
	"""
	replace given coordiantes in an image to the mean colour

	args:
		img (pytorch tensor): input image
		coords (numpy ndarray): coordinates to replace in the input image
		device (string): pytorch device e.g. cpu or cuda:0

	return:
		img (pytorch tensor)
	"""

	# get image pixel values
	mean_0 = img[0].mean()
	mean_1 = img[1].mean()
	mean_2 = img[2].mean()

	# replace pixels with mean
	I,J = np.transpose(coords)
	img[:, torch.from_numpy(I).to(device), torch.from_numpy(J).to(device)] = torch.reshape(torch.tensor([mean_0, mean_1, mean_2]).to(device), (3,1)).expand(3, len(coords))

	return img


def combine_and_order(saliency_maps, concept_points=None):
	"""
	Combine saliency maps and order the corrdinates of each pixel from most relevant to least. If concept_points is set to None
	then the entire saliency map will be used, otherwise only the relevancy inside the concept regions will be used.

	args:
		saliency_maps (list of pytorch tensors): list of saliency maps
		concept_points (list of pytorch tensors): list of cocnept locations in the order top left, top right, bottom left, bottom right.

	return:
		all_cords (numpy ndarray)
	"""

	# only keep saliency maps in concept locations
	if concept_points != None:
		final_saliency = np.empty(saliency_maps[0].squeeze(0).permute(1, 2, 0).cpu().numpy().shape)
		final_saliency.fill(np.nan)

		# compute the slope at the top of the card (top left and top right coords)
		dY = concept_points[0][0][1] - concept_points[0][1][1]
		dX = concept_points[0][0][0] - concept_points[0][1][0]
		angle = (np.degrees(np.arctan2(dY, dX)) - 0).item()

		# rotate round the center of the input
		rotateCenter = ((final_saliency.shape[0] // 2, final_saliency.shape[1] // 2))

		height, width = final_saliency.shape[:2]

		final_saliency = transform_points(final_saliency, 1, rotateCenter, angle, w_h=(width, height), image=True)

		for concept_points_index in range(len(concept_points)):
			points = concept_points[concept_points_index] # top left, top right, bottom left, bottom right

			img_saliency = saliency_maps[concept_points_index].squeeze(0).permute(1, 2, 0).cpu().numpy()

			img_saliency = transform_points(img_saliency, 1, rotateCenter, angle, w_h=(width, height), image=True)
			points = transform_points(points.cpu().numpy(), 1, rotateCenter, angle, w_h=(width, height))

			# if concept is rotated by 180 degrees then revse points
			if img_saliency[points[3][1]:points[0][1], points[3][0]:points[0][0]].shape[:2] == (0,0):
				final_saliency[points[0][1]:points[3][1], points[0][0]:points[3][0]] = img_saliency[points[0][1]:points[3][1], points[0][0]:points[3][0]] # top left (x):bottom left (x), top left (y):top right (y)
			else:
				final_saliency[points[3][1]:points[0][1], points[3][0]:points[0][0]] = img_saliency[points[3][1]:points[0][1], points[3][0]:points[0][0]] # top left (x):bottom left (x), top left (y):top right (y)

		rotateCenter = ((final_saliency.shape[0] // 2, final_saliency.shape[1] // 2))

		final_saliency = transform_points(final_saliency, 1, rotateCenter, -angle, w_h=(width, height), image=True)

		# center crop to remove black bars
		img_center = ((final_saliency.shape[0] // 2, final_saliency.shape[1] // 2))
		final_size = (299, 299)
		final_saliency = final_saliency[int(img_center[0]-final_size[0]//2):int(img_center[0]+final_size[0]//2), int(img_center[1]-final_size[1]//2):int(img_center[1]+final_size[1]//2)]

		# combine saliency maps
		sum_gradient = np.sum(final_saliency, 2)
		attr = cv2.resize(sum_gradient, dsize=(700, 700), interpolation=cv2.INTER_AREA)

		# order relevance
		attr_sort = (-attr).argsort(axis=None, kind='mergesort')
		all_cords = np.unravel_index(attr_sort, attr.shape)
		all_cords = np.vstack(all_cords).T
		return all_cords[:-np.count_nonzero(np.isnan(attr))]
	else:
		# combine saliency maps
		sum_gradient = torch.sum(torch.cat(saliency_maps), 0)
		sum_gradient = sum_gradient.unsqueeze(0)  # resize to orginal image size
		attr = heatmap(sum_gradient, cmap_name='seismic', reduce=True)
		attr = cv2.resize(attr.squeeze(), dsize=(700, 700), interpolation=cv2.INTER_AREA)


		# order relevance
		attr_sort = (-attr).argsort(axis=None, kind='mergesort')
		all_cords = np.unravel_index(attr_sort, attr.shape)
		all_cords = np.vstack(all_cords).T
		return all_cords


def transform_points(input, scale, rotateCenter, angle, w_h=None, image=False):
	"""
	transform points or image

	args:
		input (numpy ndarray): image or points to transform
		scale (float): scale factor. setting to 1 will keep the input scale
		rotateCenter (tuple of ints): coordinates to rotate the input round
		angle (float): angle to rotate input
		w_h (tuple of ints): width and height of image being scaled (or image points are for)
		image (bool): if True return warpAffine, else return dot product

	return:
		numpy ndarray
	"""

	# grab the rotation matrix for rotating and scaling the face
	M = cv2.getRotationMatrix2D(rotateCenter, angle, scale)

	if w_h != None:
		# rotation calculates the cos and sin, taking absolutes of those.
		abs_cos = abs(M[0,0])
		abs_sin = abs(M[0,1])

		# find the new width and height bounds
		bound_w = int(w_h[1] * abs_sin + w_h[0] * abs_cos)
		bound_h = int(w_h[1] * abs_cos + w_h[0] * abs_sin)

		# translate card to align center the top of the card to be in the center of the output
		M[0, 2] += bound_w/2 - rotateCenter[0]
		M[1, 2] += bound_h/2 - rotateCenter[1]

	# we are transforming an entire image / array
	if image:
		return cv2.warpAffine(input, M, (int(bound_w), int(bound_h)),
			flags=cv2.INTER_CUBIC)
	# we are transforming an array of points
	else:
		# apply transform to card coordinates
		ones = np.ones(shape=(len(input), 1))
		points_ones = np.hstack([input, ones])
		# transform points
		return M.dot(points_ones.T).T.astype(int)


def gen_random(data_loader, threshold, save_path, concept_only=False, device="cpu"):
	"""
	args:
		data_loader (pytorch DataLoader): Dataloader for dataset
		threshold (float): float between 0 and 1 to set how many features to keep
		save_path (string): path to save transformed images
		concept_only (bool): If True features will only be removed from concepts, if False features will be removed from entire inputs
		device (string): pytorch device e.g. cpu or cuda:0
	"""
	for images, labels, concepts, extra in tqdm(data_loader, unit="batch"):
		paths = extra["path"]
		card_points = extra["points"]
		orig_imgs = extra["orig_img"]
		for image_index in range(images.size(0)):
			image = images[image_index].to(device)
			orig_img = orig_imgs[image_index].to(device)
			path = paths[image_index]

			# only get coordinates inside concept areas
			if concept_only:
				all_cords = []
				for concept_points_index in range(card_points[image_index].size(0)):
					points = card_points[image_index][concept_points_index] # top left, top right, bottom left, bottom right

					# compute the slope at the top of the card (top left and top right coords)
					dY = points[0][1] - points[1][1]
					dX = points[0][0] - points[1][0]
					angle = (np.degrees(np.arctan2(dY, dX)) - 90).item()

					# determine how much to scale the card points to align them with orig_img
					scale = orig_img.size(1) / image.size(1)

					rotateCenter = ((0,0))

					points = transform_points(points.cpu().numpy(), scale, rotateCenter, angle)

					y_points = range(points[1][1], points[0][1])
					x_points = range(points[0][0], points[2][0])

					card_cords = list(itertools.product(x_points, y_points))

					card_cords = transform_points(np.array(card_cords), 1, rotateCenter, -angle)

					# add points to list. sway x and y as they are the wrong way round
					all_cords.extend([point[::-1] for point in card_cords.tolist()])

			# get coordinates for entire input
			else:
				y_points = range(0, orig_img.size(1))
				x_points = range(0, orig_img.size(2))
				all_cords = list(itertools.product(x_points, y_points))

			random.shuffle(all_cords)

			replace_num = int(len(all_cords) * threshold)
			orig_img = replace_fetures(orig_img, all_cords[:replace_num], device=device)

			if not os.path.exists(f'{save_path}'):
				os.makedirs(f'{save_path}')
			save_image(orig_img, f'{save_path}/{path.split("/")[-1][:-4]}.png')


def gen_lrp(model_parts, rules, data_loader, threshold, save_path, concept_only=False, device="cpu"):
	"""
	args:
		model (list pytorch model): Model used to generate features to remove
		rules (list of strings): KRP rules to apply to model parts
		data_loader (pytorch DataLoader): Dataloader for dataset
		threshold (float): float between 0 and 1 to set how many features to keep
		save_path (string): path to save transformed images
		concept_only (bool): If True features will only be removed from concepts, if False features will be removed from entire inputs
		device (string): pytorch device e.g. cpu or cuda:0
	"""
	for images, labels, concepts, extra in tqdm(data_loader, unit="batch"):
		paths = extra["path"]
		orig_imgs = extra["orig_img"]
		card_points = extra["points"]
		card_classes = extra["card_classes"]
		for image_index in range(images.size(0)):
			orig_img = orig_imgs[image_index].to(device)
			path = paths[image_index]
			cards = card_classes[image_index]
			image = images[image_index].to(device)
			image = image.unsqueeze(0)
			image.requires_grad_(True)
			image.grad = None  # Reset gradient

			pred_concepts = image
			for idx, model_part in enumerate(model_parts):
				pred_concepts = model_part.forward(pred_concepts, explain=True, rule=rules[idx])

			sorted_concept, indices = torch.sort(pred_concepts, descending=True)

			# get top 3 predicted concepts saliency maps
			concept_saliency = []
			concept_points = []
			for i in range(cards.size(0)):

				filter_out = torch.zeros_like(pred_concepts)
				filter_out[:, cards[i].item()] += 1

				# Get the gradient of each input
				image_gradient = torch.autograd.grad(
					pred_concepts,
					image,
					grad_outputs=filter_out,
					retain_graph=True)[0]

				concept_saliency.append(image_gradient)
				concept_points.append(card_points[image_index][i])

			# only get coordinates inside concept areas
			if concept_only:
				all_cords = combine_and_order(concept_saliency, concept_points)
			# get coordinates over entire input
			else:
				all_cords = combine_and_order(concept_saliency, None)

			# remove features
			replace_num = int(len(all_cords) * threshold)
			orig_img = replace_fetures(orig_img, all_cords[:replace_num], device=device)

			# save modified image
			if not os.path.exists(f'{save_path}'):
				os.makedirs(f'{save_path}')
			save_image(orig_img, f'{save_path}/{path.split("/")[-1][:-4]}.png')


def gen_sg_sq(model, data_loader, threshold, save_path, device="cpu"):
	"""
	args:
		model (pytorch model): Model used to generate features to remove
		data_loader (pytorch DataLoader): Dataloader for dataset
		threshold (float): float between 0 and 1 to set how many features to keep
		save_path (string): path to save transformed images
		device (string): pytorch device e.g. cpu or cuda:0
	"""
	for images, labels, concepts, extra in tqdm(data_loader, unit="batch"):
		paths = extra["path"]
		orig_imgs = extra["orig_img"]
		for image_index in range(images.size(0)):
			orig_img = orig_imgs[image_index].to(device)
			path = paths[image_index]
			image = images[image_index].to(device)
			image = image.unsqueeze(0)
			image.requires_grad_(True)
			image.grad = None  # Reset gradient
			baselines = image * 0

			pred_concepts = model(image)

			sorted_concept, indices = torch.sort(pred_concepts, descending=True)

			ig = IntegratedGradients(model)
			nt = NoiseTunnel(ig)

			# get top 3 predicted concepts saliency maps
			concept_saliency = []
			for i in range(3):

				attributions_ig = nt.attribute(
					image,
					target=indices[0][i],
					baselines=baselines,
					nt_type='smoothgrad_sq',
					nt_samples=25,
					internal_batch_size=25,
					stdevs=0.2
				)

				concept_saliency.append(attributions_ig)

			all_cords = combine_and_order(concept_saliency)

			# remove features
			replace_num = int((700 * 700) * threshold)
			orig_img = replace_fetures(orig_img, all_cords[:replace_num], device=device)

			# save modified image
			if not os.path.exists(f'{save_path}'):
				os.makedirs(f'{save_path}')
			save_image(orig_img, f'{save_path}/{path.split("/")[-1][:-4]}.png')


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Evaluate a Concept Bottleneck Model with ROAR')
	parser.add_argument(
		'--batch_size',
		type=int,
		default=16,
		help='Number of samples per batch (default: 16)'
	)
	parser.add_argument(
		'--weights_dir',
		type=str,
		help='File path to pytorch weight file'
	)
	parser.add_argument(
		'--dataset_root',
		type=str,
		help='Path to dataset root directory'
	)
	parser.add_argument(
		'--dataset_postfix',
		type=str,
		default="",
		help='Path to dataset split files from dataset root (default: "")'
	)
	parser.add_argument(
		'--dataset_split',
		type=str,
		help='Name of the dataset version to save'
	)
	parser.add_argument(
		'--pretrained_weights',
		type=str,
		help='File path to pytorch weight file'
	)
	parser.add_argument(
		'--n_classes',
		type=int,
		help='Number of classes in dataset'
	)
	parser.add_argument(
		'--n_concepts',
		type=int,
		help='Number of concepts/attributes in dataset'
	)
	parser.add_argument(
		'--use_sig',
		action='store_true',
		help='Use sigmoid between the XtoC and the CtoY model parts'
	)
	parser.add_argument(
		'--use_cpu',
		action='store_true',
		help='Only use the system CPU, even if CUDA is available'
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.use_cpu) else "cpu")
	if torch.cuda.is_available() and not args.use_cpu:
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	train_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", args.batch_size, dataset_root=args.dataset_root, num_workers=1, return_path=True, return_orig=True, resize_orig=False, drop_last=False, train_transform=False, return_points=True)
	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", args.batch_size, dataset_root=args.dataset_root, num_workers=1, return_path=True, return_orig=True, resize_orig=False, drop_last=False, train_transform=False, return_points=True)

	model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, use_sigmoid=args.use_sig, use_modulelist=False)
	model.load_state_dict(torch.load(f"{args.pretrained_weights}", map_location=torch.device('cpu')))
	model_name = args.pretrained_weights.split("/")[-1][:-4]

	XtoC = convert(model.x_to_c_model)

	XtoC_model_1 = []
	XtoC_model_2 = []
	XtoC_model_3 = []

	for idx, m in enumerate(XtoC.children()):
		if idx < 15:
			XtoC_model_1.append(m)
		elif idx < 29:
			XtoC_model_2.append(m)
		else:
			XtoC_model_3.append(m)

	XtoC_model_1 = lrp.Sequential(*XtoC_model_1)
	XtoC_model_2 = lrp.Sequential(*XtoC_model_2)
	XtoC_model_3 = lrp.Sequential(*XtoC_model_3)

	XtoC_model_1.to(device).eval()
	XtoC_model_2.to(device).eval()
	XtoC_model_3.to(device).eval()

	# Generate dataset versions

	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]  # percentage of features (pixels) to remove

	## random

	for threshold in thresholds:
		save_path = f'dataset/roar/random/{threshold}/imgs/{args.dataset_split}'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_random(train_loader, threshold, save_path=save_path, device=device)
			gen_random(val_loader, threshold, save_path=save_path, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')
		copy_tree(f'dataset/playing-cards/splits', f"dataset/roar/random/{threshold}/splits")

	print("Completed random features")

	## LPR
	"""
	for threshold in thresholds:
		save_path = f'dataset/roar/lrp/{model_name}/{threshold}/imgs/{args.dataset_split}'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], train_loader, threshold, save_path=save_path, device=device)
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], val_loader, threshold, save_path=save_path, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')
		copy_tree(f'dataset/playing-cards/splits', f'dataset/roar/lrp/{model_name}/{threshold}/splits')

	print("Completed LRP features")

	## LPR (only positive relevance)

	for threshold in thresholds:
		save_path = f'dataset/roar/lrp_positive/{model_name}/{threshold}/imgs/{args.dataset_split}'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "alpha2beta1", "alpha2beta1"], train_loader, threshold, save_path=save_path, device=device)
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "alpha2beta1", "alpha2beta1"], val_loader, threshold, save_path=save_path, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')
		copy_tree(f'dataset/playing-cards/splits', f'dataset/roar/lrp_positive/{model_name}/{threshold}/splits')

	print("Completed LRP (positive) features")
	"""

	## SG_SQ

	"""
	XtoC_model = model.x_to_c_model.to(device).eval()

	for threshold in thresholds:
		save_path = f'dataset/roar/sg_sq/{model_name}/{threshold}/imgs/{args.dataset_split}'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_sg_sq(XtoC_model, train_loader, threshold, save_path=save_path, device=device)
			gen_sg_sq(XtoC_model, val_loader, threshold, save_path=save_path, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')

	print("Completed smoothgrad squared features")
	"""

	## concept_removal

	for threshold in thresholds:
		save_path = f'dataset/roar/concept_random/{threshold}/imgs/{args.dataset_split}'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_random(train_loader, threshold, save_path=save_path, concept_only=True, device=device)
			gen_random(val_loader, threshold, save_path=save_path, concept_only=True, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')
		copy_tree(f'dataset/playing-cards/splits', f'dataset/roar/concept_random/{threshold}/splits')

	print("Completed concept (random) features")

	## LPR
	"""
	for threshold in thresholds:
		save_path = f'dataset/roar/concept_lrp/{model_name}/{threshold}/imgs/{args.dataset_split}'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], train_loader, threshold, save_path=save_path, concept_only=True, device=device)
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], val_loader, threshold, save_path=save_path, concept_only=True, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')
		copy_tree(f'dataset/playing-cards/splits', f'dataset/roar/concept_lrp/{model_name}/{threshold}/splits')

	print("Completed concept (LRP) features")

	## LPR (only positive relevance)

	for threshold in thresholds:
		save_path = f'dataset/roar/concept_lrp_positive/{model_name}/{threshold}/imgs/{args.dataset_split}'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "alpha2beta1", "alpha2beta1"], train_loader, threshold, save_path=save_path, concept_only=True, device=device)
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "alpha2beta1", "alpha2beta1"], val_loader, threshold, save_path=save_path, concept_only=True, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')
		copy_tree(f'dataset/playing-cards/splits', f'dataset/roar/concept_lrp_positive/{model_name}/{threshold}/splits')

	print("Completed concept (LRP positive) features")
	"""
