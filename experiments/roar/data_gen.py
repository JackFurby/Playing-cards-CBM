import torch
import argparse
from tqdm import tqdm
import itertools
from matplotlib import pyplot as plt
import random
import os
from dataset.dataset import load_data
from torchvision.utils import save_image
from models.model import End2EndModel
from models.lrp_converter import convert
import lrp
from experiments.utils import heatmap
import cv2
from captum.attr import IntegratedGradients, NoiseTunnel

import time
import numpy as np


def replace_fetures(img, coords):
	"""
	"""

	# get image pixel values
	mean_0 = img[0].mean()
	mean_1 = img[1].mean()
	mean_2 = img[2].mean()

	# replace pixels with mean
	for coord in coords:
		img[0][coord[0]][coord[1]] = mean_0
		img[1][coord[0]][coord[1]] = mean_1
		img[2][coord[0]][coord[1]] = mean_2

	return img


def combine_and_order(saliency_maps):
	"""
	"""
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


def gen_random(data_loader, threshold, save_path, device="cpu"):
	for images, labels, concepts, extra in tqdm(data_loader, unit="batch"):
		paths = extra["path"]
		orig_imgs = extra["orig_img"]
		for image_index in range(images.size(0)):
			image = orig_imgs[image_index].to(device)
			path = paths[image_index]

			y_points = range(0, image.size(1))
			x_points = range(0, image.size(2))
			all_cords = list(itertools.product(x_points, y_points))
			random.shuffle(all_cords)

			replace_num = int(len(all_cords) * threshold)
			image = replace_fetures(image, all_cords[:replace_num])

			if not os.path.exists(f'{save_path}'):
				os.makedirs(f'{save_path}')
			save_image(image, f'{save_path}/{path.split("/")[-1][:-4]}.png')


def gen_lrp(model_parts, rules, data_loader, threshold, save_path, device="cpu"):
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

			pred_concepts = image
			for idx, model_part in enumerate(model_parts):
				pred_concepts = model_part.forward(pred_concepts, explain=True, rule=rules[idx])

			sorted_concept, indices = torch.sort(pred_concepts, descending=True)

			# get top 3 predicted concepts saliency maps
			concept_saliency = []
			for i in range(3):

				filter_out = torch.zeros_like(pred_concepts)
				filter_out[:, indices[0][i].item()] += 1

				# Get the gradient of each input
				image_gradient = torch.autograd.grad(
					pred_concepts,
					image,
					grad_outputs=filter_out,
					retain_graph=True)[0]

				concept_saliency.append(image_gradient)

			all_cords = combine_and_order(concept_saliency)

			# remove features
			replace_num = int((700 * 700) * threshold)
			orig_img = replace_fetures(orig_img, all_cords[:replace_num])

			# save modified image
			if not os.path.exists(f'{save_path}'):
				os.makedirs(f'{save_path}')
			save_image(orig_img, f'{save_path}/{path.split("/")[-1][:-4]}.png')


def gen_sg_sq(model, data_loader, threshold, save_path, device="cpu"):
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
			orig_img = replace_fetures(orig_img, all_cords[:replace_num])

			# save modified image
			if not os.path.exists(f'{save_path}'):
				os.makedirs(f'{save_path}')
			save_image(orig_img, f'{save_path}/{path.split("/")[-1][:-4]}.png')


def gen_concept_removal(data_loader, threshold, save_path, device="cpu"):
	pass


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

	train_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", args.batch_size, dataset_root=args.dataset_root, resampling=True, num_workers=1, return_path=True, return_orig=True, resize_orig=False, drop_last=False)
	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", args.batch_size, dataset_root=args.dataset_root, num_workers=1, return_path=True, return_orig=True, resize_orig=False, drop_last=False)

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
		save_path = f'dataset/roar/random/{threshold}/imgs'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_random(train_loader, threshold, save_path=save_path, device=device)
			gen_random(val_loader, threshold, save_path=save_path, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')

	print("Completed random features")

	## LPR

	for threshold in thresholds:
		save_path = f'dataset/roar/lrp/{model_name}/{threshold}/imgs'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], train_loader, threshold, save_path=save_path, device=device)
			gen_lrp([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], val_loader, threshold, save_path=save_path, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')

	print("Completed LRP features")

	## SG_SQ

	XtoC_model = model.x_to_c_model.to(device).eval()

	for threshold in thresholds:
		save_path = f'dataset/roar/sg_sq/{model_name}/{threshold}/imgs'
		if not os.path.exists(save_path):
			print(f'Generating images for threshold {threshold}')
			gen_sg_sq(XtoC_model, train_loader, threshold, save_path=save_path, device=device)
			gen_sg_sq(XtoC_model, val_loader, threshold, save_path=save_path, device=device)
		else:
			print(f'Images for threshold {threshold} already exist')

	print("Completed smoothgrad squared features")

	## concept_removal
