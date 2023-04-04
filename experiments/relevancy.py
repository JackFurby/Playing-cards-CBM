import torch
import numpy as np
import argparse
from torchmetrics import F1Score
from dataset.dataset import load_data
from dataset.utils import IndexToString
from torchvision.utils import save_image
#from models.model import End2EndModel
from models.vgg import End2EndModel
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from experiments.utils import heatmap, project
import lrp
from models.lrp_converter import convert
import cv2


# A single figure is defined as putting this in a loop quickly uses up all system memory.
# This figure is reused for all plots
FIG, AX = plt.subplots()


def extract_card(image_gradient, card_cord):
	"""
	1. rotate card and corrdinates
	2. extract card at the given corrdinates
	3. return the extracted card as a tensor
	"""

	card_saliency = image_gradient[0].permute(1, 2, 0).cpu().numpy()

	height, width = card_saliency.shape[:2]

	# compute the slope at the top of the card (top left and top right coords)
	dY = card_cord[0][1] - card_cord[1][1]
	dX = card_cord[0][0] - card_cord[1][0]
	angle = (np.degrees(np.arctan2(dY, dX)) - 180).item()

	# determine the scale of the card after transformation by taking
	# the ratio of the distance between top points in the current
	# image to the ratio of desired distance
	dist = np.sqrt((dX ** 2) + (dY ** 2))
	scale = 1

	# compute the point to rotate the card round - center of top edge
	rotateCenter = ((card_cord[0][0].item() + card_cord[1][0].item()) // 2,
		(card_cord[0][1].item() + card_cord[1][1].item()) // 2)

	# grab the rotation matrix for rotating and scaling the face
	M = cv2.getRotationMatrix2D(rotateCenter, angle, scale)

	# rotation calculates the cos and sin, taking absolutes of those.
	abs_cos = abs(M[0,0])
	abs_sin = abs(M[0,1])

	# find the new width and height bounds
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)

	# translate card to align center the top of the card to be in the center of the output
	M[0, 2] += bound_w/2 - rotateCenter[0]
	M[1, 2] += bound_h/2 - rotateCenter[1]

	# apply the affine transformation
	output = cv2.warpAffine(card_saliency, M, (int(bound_w), int(bound_h)),
		flags=cv2.INTER_CUBIC)

	# apply transform to card coordinates
	card_cord = card_cord.cpu().numpy()
	ones = np.ones(shape=(len(card_cord), 1))
	points_ones = np.hstack([card_cord, ones])
	# transform points
	card_cord = M.dot(points_ones.T).T.astype(int)

	output = output[card_cord[0][1]:card_cord[2][1], card_cord[0][0]:card_cord[1][0], :]  # top left (x):bottom left (x), top left (y):top right (y)

	# get card positive and negative relevance (cannot do after scale)
	neg = np.sum(output.clip(max=0))  # sum card negative relevance
	pos = np.sum(output.clip(min=0))  # sum card positive relevance

	output = cv2.resize(output, (346, 519), interpolation=cv2.INTER_AREA)

	output = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0)

	return output, pos, neg


def gen_lrp_saliency(model_parts, rules, data_loader, concept_index_to_string, device="cpu", model_name="model", target_concept=None, samples_per_class=-1):
	"""
	model_parts: List of pytorch models
	rules: List of LRP rules
	data_loader: Pytorch dataloader
	concept_index_to_string: IndexToString object
	device: string pytorch device
	model_name: name of model (used for saving results to a recognisable file path)
	target_concept: None if results are for entire samples, int of concept index if averaging concept saliency maps
	samples_per_class: Number of samples to generate saliency maps for (ignored if set to -1)
	"""
	# keep track of number of samples we have generated results for
	sample_gen_count = []
	for i in range(data_loader.dataset.num_classes):
		sample_gen_count.append(0)

	sample_num = 0
	concept_saliency, card_relevance, all_relevance, all_cards_relevance = [], [], [], []
	for images, labels, concepts, extra in tqdm(data_loader, unit="batch"):
		card_points = extra["points"]
		card_classes = extra["card_classes"]
		orig_image = extra["orig_img"]
		for image_index in range(images.size(0)):
			image = images[image_index].to(device)
			image = image.unsqueeze(0)
			image.requires_grad_(True)
			image.grad = None  # Reset gradient

			sample_gen_count[labels[image_index].item()] += 1
			if samples_per_class == -1 or sample_gen_count[labels[image_index].item()] <= samples_per_class:

				pred_concepts = image
				for idx, model_part in enumerate(model_parts):
					pred_concepts = model_part.forward(pred_concepts, explain=True, rule=rules[idx])

				if target_concept == None:
					save_dir = f"results/lrp/{model_name}/{labels[image_index].item()}/{sample_num}/"
					if not os.path.exists(f"{save_dir}"):
						os.makedirs(f"{save_dir}")

					# Save input image
					save_image(orig_image[image_index], f'{save_dir}/input.png')

					# Write true concepts to file
					with open(f'{save_dir}/true-concepts.txt', mode='wt', encoding='utf-8') as f:
						f.write("True concepts for sample\n")
						f.write("========================\n")
						for idy, concept in enumerate(concepts[image_index]):
							f.write(f"{idy}: {concept_index_to_string(idy)}, concept value: {concept.item()}\n")

					# Write concept predictions to file (sorted by concept value)
					pred_concepts_readable = torch.nn.Sigmoid()(pred_concepts)
					pred_list = []
					for idx, i in enumerate(pred_concepts_readable[0]):
						pred_list.append((idx, i.item()))
					pred_concepts_readable = sorted(pred_list, key=lambda x: -x[1])
					with open(f'{save_dir}/concept-pred.txt', mode='wt', encoding='utf-8') as f:
						f.write("Predicted concepts (with Sigmoid)\n")
						f.write("=================================\n")
						for item in pred_concepts_readable:
							f.write(f"{item[0]}: {concept_index_to_string(item[0])}, Sig value: {item[1]}\n")

				for i in range(pred_concepts.size(1)):
					concept_pred_present = torch.nn.Sigmoid()(pred_concepts[0][i]).item() > 0.5  # only avg cards predicted as present
					current_concept_target = i == target_concept  # only avg target concepts
					true_concept_present = concepts[image_index][i].item() == 1  # only avg concepts present in concept labels
					# only continue if generating saliency map for entire sample or if target concept if correctly predicted as present
					if (target_concept == None) or (concept_pred_present and current_concept_target and true_concept_present):
						filter_out = torch.zeros_like(pred_concepts)
						filter_out[:,i] += 1

						# Get the gradient of each input
						image_gradient = torch.autograd.grad(
							pred_concepts,
							image,
							grad_outputs=filter_out,
							retain_graph=True)[0]

						# generating saliency map for entire sample
						if target_concept == None:

							attr = heatmap(image_gradient, cmap_name='seismic')

							# Save concept saliency map
							AX.clear()
							AX.set_axis_off()
							AX.imshow(attr.squeeze(), cmap='seismic')
							FIG.savefig(f'{save_dir}{concept_index_to_string(i)}.png', bbox_inches='tight', pad_inches = 0)

						# computing relevancy in and outside of concepts
						else:

							if card_classes[image_index].size(0) == 1:  # single playing card in image
								card_cord = card_points[image_index][0]
							else:  # multiple playing cards in image, concepts match card_classes
								for concept in range(card_classes[image_index].size(0)):
									if card_classes[image_index][concept].item() == i:  # true if card in image matches i
										card_cord = card_points[image_index][concept]

							for i in range(card_classes[image_index].size(0)):
								card_gradient, card_pos, card_neg = extract_card(image_gradient, card_points[image_index][i])
								all_cards_relevance.append((card_pos, card_neg))

							card_gradient, card_pos, card_neg = extract_card(image_gradient, card_cord)
							card_relevance.append((card_pos, card_neg))

							all_pos = torch.clip(image_gradient, min=0)  # all image positive relevance
							all_neg = torch.clip(image_gradient, max=0)  # all image negative relevance
							all_relevance.append((torch.sum(all_pos).item(), torch.sum(all_neg).item()))

							concept_saliency.append(card_gradient)

			sample_num += 1

	if len(concept_saliency) > 0:
		avg_gradient = torch.mean(torch.cat(concept_saliency), 0)
		avg_gradient = avg_gradient.unsqueeze(0)
		attr = heatmap(avg_gradient, cmap_name='seismic')

		AX.clear()
		AX.set_axis_off()
		AX.imshow(attr.squeeze(), cmap='seismic')
		FIG.savefig(f'results/lrp/{model_name}/average_{concept_index_to_string(target_concept)}.png', bbox_inches='tight', pad_inches = 0)

	if target_concept != None:
		# work out the proportion of positive and negative relevance for the concept from the total relevance
		concept_pos = [x[0] for x in card_relevance]
		concept_neg = [x[1] for x in card_relevance]
		all_pos = [x[0] for x in all_relevance]
		all_neg = [x[1] for x in all_relevance]
		cards_pos = [x[0] for x in all_cards_relevance]
		cards_neg = [x[1] for x in all_cards_relevance]

		concept_all_pos_prop = sum(concept_pos) / sum(all_pos)
		concept_all_neg_prop = sum(concept_neg) / sum(all_neg)

		concept_cards_pos_prop = sum(concept_pos) / sum(cards_pos)
		concept_cards_neg_prop = sum(concept_neg) / sum(cards_neg)

		return concept_all_pos_prop, concept_all_neg_prop, concept_cards_pos_prop, concept_cards_neg_prop
	# We only use the return values if we have a target concept
	else:
		return None


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
		'--use_sig',
		action='store_true',
		help='Use sigmoid between the XtoC and the CtoY model parts'
	)
	parser.add_argument(
		'--use_cpu',
		action='store_true',
		help='Only use the system CPU, even if CUDA is available'
	)
	parser.add_argument(
		'--samples_per_class',
		type=int,
		help='Max number of samples to generate results for per class in the dataset (set to -1 to ignore)',
		default=25
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.use_cpu) else "cpu")
	if torch.cuda.is_available() and not args.use_cpu:
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", args.batch_size, dataset_root=args.dataset_root, num_workers=1, return_points=True, return_orig=True)
	concept_index_to_string = IndexToString(f"./{args.dataset_root}/{args.dataset_postfix}/concepts.txt")

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

	# gen saliency maps for full samples
	gen_lrp_saliency([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], val_loader, concept_index_to_string, device, model_name, target_concept=None, samples_per_class=args.samples_per_class)

	concept_relevance_prop = []  # all_pos, all_neg, cards_pos, cards_neg
	# gen average saliency maps for concepts
	for i in range(len(concept_index_to_string.string_dict)):
		concept_all_pos_prop, concept_all_neg_prop, concept_cards_pos_prop, concept_cards_neg_prop = gen_lrp_saliency([XtoC_model_1, XtoC_model_2, XtoC_model_3], ["alpha1beta0", "epsilon", "gradient"], val_loader, concept_index_to_string, device, model_name, target_concept=i)
		concept_relevance_prop.append((concept_all_pos_prop, concept_all_neg_prop, concept_cards_pos_prop, concept_cards_neg_prop))

	# open file in write mode
	with open(f"results/lrp/{model_name}/concept_props.txt", "w") as fp:
		fp.write(f"concept ID, concept string, positive proportion to entire image (%), negative proportion to entire image (%), positive proportion to all cards (%), negative proportion to all cards (%)\n")
		for idx, item in enumerate(concept_relevance_prop):
			# write each item on a new line
			fp.write(f"{idx}, {concept_index_to_string(idx)}, {item[0]}, {item[1]}, {item[2]}, {item[3]}\n")
