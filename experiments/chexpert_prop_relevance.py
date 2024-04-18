import torch
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data
from dataset.utils import IndexToString
from torchvision.utils import save_image
from models.vgg import End2EndModel
import models.chexpert.classifier as classifier
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from experiments.utils import heatmap, project
import lrp
from models.lrp_converter import convert
import cv2
from easydict import EasyDict as edict
import json
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import pickle

from captum.attr import IntegratedGradients, Saliency, GuidedGradCam
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from dataset.utils import transform as chexpert_transform
seg_transform = chexpert_transform


# A single figure is defined as putting this in a loop quickly uses up all system memory.
# This figure is reused for all plots
FIG, AX = plt.subplots()

base_cmap = pl.cm.seismic
seismic_alpha = base_cmap(np.arange(base_cmap.N))
seismic_alpha[:,-1] = np.concatenate((np.ones(115), (np.logspace(1, 0, 13)/10), (np.logspace(0, 1, 13)/10), np.ones(115)), axis=0)  # set alpha
seismic_alpha = ListedColormap(seismic_alpha)  # create new colormap

base_cmap = pl.cm.Greens
green_alpha = base_cmap(np.arange(base_cmap.N))
green_alpha[:,-1] = np.linspace(0, 1, 256)  # set alpha
green_alpha = ListedColormap(green_alpha)  # create new colormap

f = open('./dataset/chexpert/CheXlocalize/gt_annotations_val.json')
segmentations = json.load(f)
f.close()
f = open('./dataset/chexpert/CheXlocalize/gt_annotations_test.json')
segmentations_test = json.load(f)
f.close()
segmentations.update(segmentations_test)


def get_human_segmentation(sample_key, concept_key, segmentations=segmentations):

	if sample_key not in segmentations.keys():
		return None
	if concept_key not in segmentations[sample_key].keys():
		return None

	polygons = segmentations[sample_key][concept_key]

	poly = Image.new('1', (segmentations[sample_key]["img_size"][1], segmentations[sample_key]["img_size"][0]))
	for polygon in polygons:
		coords = [(point[0], point[1]) for point in polygon]
		ImageDraw.Draw(poly).polygon(coords,  outline=1, fill=1)

	return poly


def gen_relevancy_prop(model, mode, data_loader, concept_index_to_string, target_concept, device="cpu", model_name="model"):
	"""
	model: pytorch model
	mode: string captum attribute name
	data_loader: Pytorch dataloader
	concept_index_to_string: IndexToString object
	target_concept: Concept index
	device: string pytorch device
	model_name: name of model (used for saving results to a recognisable file path)
	"""

	if mode in ["ig_sg", "ig_sg_sq"]:
		ig = IntegratedGradients(model)
		nt = NoiseTunnel(ig)
	elif mode == "gradCAM":
		guided_gc = GuidedGradCam(model, model.backbone.features.denseblock4.denselayer16[5])

	concept_relevance, all_relevance, concept_predictions = [], [], []
	for images, labels, concepts, extra in tqdm(data_loader, unit="batch"):
		for image_index in range(images.size(0)):  # for each image
			image = images[image_index].to(device)
			true_concepts = concepts[image_index]
			image = image.unsqueeze(0)
			image.requires_grad_(True)
			image.grad = None  # Reset gradient

			if mode in ["ig_sg", "ig_sg_sq"]:
				baselines = image * 0

			pred_concepts = model(image)

			for sample in pred_concepts:
				for i, concept in enumerate(sample):
					if (i == target_concept) and (true_concepts[i] == 1):  # Only get saliency for target concept. ground truth also needs to have present concept

						segmentation_key = "_".join(extra["path"][image_index].split("/")[-3:]).split(".")[0]

						# get ground truth segmentation
						if image_index == 0:
							concept_key = "Enlarged Cardiomediastium"
						if image_index == 1:
							concept_key = "Cardiomegaly"
						if image_index == 2:
							concept_key = "Airspace Opacity"  # Lung Opacity
						if image_index == 3:
							concept_key = "Lung Lesion"
						if image_index == 4:
							concept_key = "Edema"
						if image_index == 5:
							concept_key = "Consolidation"
						if image_index == 6:
							concept_key = "Pneumonia"
						if image_index == 7:
							concept_key = "Atelectasis"
						if image_index == 8:
							concept_key = "Pneumothorax"
						if image_index == 9:
							concept_key = "Pleural Effusion"
						if image_index == 10:
							concept_key = "Plaural Other"
						if image_index == 11:
							concept_key = "Fracture"
						if image_index == 12:
							concept_key = "Support Devices"

						segmentation_img = get_human_segmentation(segmentation_key, concept_key)

						# only generate saliency map if we have a segmentation (we need it for proportions)
						if segmentation_img != None:

							filter_out = torch.zeros_like(pred_concepts)
							filter_out[:,i] += 1

							if mode in ["ig_sg", "ig_sg_sq"]:
								if mode == "ig_sg":
									nt_type = "smoothgrad"
								else:
									nt_type = "smoothgrad_sq"

								attribution = nt.attribute(
									image,
									target=int(i),
									baselines=baselines,
									nt_type=nt_type,
									nt_samples=25,   #<<<<< chage back to 25 after testing
									internal_batch_size=25,
									stdevs=0.2
								)

							elif mode == "gradCAM":
								attribution = guided_gc.attribute(
									image,
									target=int(i),
								)

							segmentation_img = np.asarray(segmentation_img.convert("L"))
							segmentation_img = seg_transform(segmentation_img, gaussian_blur=3, use_equalizeHist=True, long_side=512, border_pad_str="mean", pixel_mean=128.0, pixel_std=64.0, resize_only=True)
							alpha = np.sum(segmentation_img, axis=-1) > 0  # only keep segmentation areas

							all_pos = torch.clip(attribution, min=0)  # all image positive relevance
							all_neg = torch.clip(attribution, max=0)  # all image negative relevance
							all_relevance.append((torch.sum(all_pos).item(), torch.sum(all_neg).item()))

							concept_pos = all_pos.detach().cpu() * alpha  # all image positive relevance
							concept_neg = all_neg.detach().cpu() * alpha  # all image negative relevance
							concept_relevance.append((torch.sum(concept_pos).item(), torch.sum(concept_neg).item()))

							concept_predictions.append(torch.nn.Sigmoid()(concept).cpu().item())

	# work out the proportion of positive and negative relevance for the concept from the total relevance
	all_pos = [x[0] for x in all_relevance]
	all_neg = [x[1] for x in all_relevance]
	concept_pos = [x[0] for x in concept_relevance]
	concept_neg = [x[1] for x in concept_relevance]

	if mode == "ig_sg_sq":
		concept_neg_prop = 0
	else:
		concept_neg_prop = sum(concept_neg) / sum(all_neg)

	concept_pos_prop = sum(concept_pos) / sum(all_pos)

	positive, negative = (concept_pos, all_pos, concept_predictions), (concept_neg, all_neg, concept_predictions)

	return concept_pos_prop, concept_neg_prop, positive, negative


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Evaluate a Concept Bottleneck Model')
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
		'--use_cpu',
		action='store_true',
		help='Only use the system CPU, even if CUDA is available'
	)
	parser.add_argument(
		'--saliency_mode',
		type=str,
		choices=["ig_sg", "ig_sg_sq", "gradCAM"],
		help='Saliency method to use'
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.use_cpu) else "cpu")
	if torch.cuda.is_available() and not args.use_cpu:
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/test.pkl", args.batch_size, dataset_root=args.dataset_root, dataset_name="chexpert", num_workers=1, return_points=True, return_orig=True, return_path=True)
	concept_index_to_string = IndexToString(f"./{args.dataset_root}/{args.dataset_postfix}/concepts.txt")

	cfg = {
		"backbone": "densenet121",
		"norm_type": "BatchNorm",
		"pretrained": True,
		"num_concepts": args.n_concepts,
		"cbm_mode": "joint",  # can be either xtoc or joint (joint is also used for sequential)
		"xtoc_only": True  # if true then only predict concepts
	}

	model = classifier.Classifier(cfg)
	model.load_state_dict(torch.load(f"{args.pretrained_weights}", map_location=torch.device('cpu'))['state_dict'], strict=True)
	model_name = args.pretrained_weights.split("/")[-1][:-4]
	model.to(device)

	concept_relevance_prop = []  # all_pos, all_neg, cards_pos, cards_neg
	positive_rel_list, negative_rel_list = [], []  # index of list is concept index
	# gen average saliency maps for concepts
	for i in range(len(concept_index_to_string.string_dict)):                                                                #<<< temp for testing
		concept_pos_prop, concept_neg_prop, positive_rel, negative_rel = gen_relevancy_prop(model, args.saliency_mode, val_loader, concept_index_to_string, i, device, model_name)
		positive_rel_list.append(positive_rel)
		negative_rel_list.append(negative_rel)
		concept_relevance_prop.append((concept_pos_prop, concept_neg_prop))

	save_dir = f"results/{args.saliency_mode}/{model_name}"
	if not os.path.exists(f"{save_dir}"):
		os.makedirs(f"{save_dir}")

	# save relevancy to pickle files
	pickle.dump(positive_rel_list, open(f"{save_dir}/positive_rel.pkl", "wb"))
	pickle.dump(negative_rel_list, open(f"{save_dir}/negative_rel.pkl", "wb"))

	# open file in write mode
	with open(f"{save_dir}/concept_props.txt", "w") as fp:
		fp.write(f"concept ID, concept string, positive proportion (%), negative proportion (%)\n")
		for idx, item in enumerate(concept_relevance_prop):
			# write each item on a new line
			fp.write(f"{idx}, {concept_index_to_string(idx)}, {item[0]}, {item[1]}\n")
