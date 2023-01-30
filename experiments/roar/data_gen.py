import torch
import argparse
from tqdm import tqdm
import itertools
from matplotlib import pyplot as plt
import random
import os
from dataset.dataset import load_data
from torchvision.utils import save_image


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

			mean_0 = image[0].mean()
			mean_1 = image[1].mean()
			mean_2 = image[2].mean()

			for coord in all_cords[:replace_num]:
				image[0][coord[0]][coord[1]] = mean_0
				image[1][coord[0]][coord[1]] = mean_1
				image[2][coord[0]][coord[1]] = mean_2

			if not os.path.exists(f'{save_path}/random_{threshold}/imgs'):
				os.makedirs(f'{save_path}/random_{threshold}/imgs')
			save_image(image, f'{save_path}/random_{threshold}/imgs/{path.split("/")[-1][:-4]}.png')


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

	train_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", args.batch_size, dataset_root=args.dataset_root, resampling=True, num_workers=1, return_path=True, return_orig=True, resize_orig=False)
	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", args.batch_size, dataset_root=args.dataset_root, num_workers=1, return_path=True, return_orig=True, resize_orig=False)

	# Generate dataset versions

	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]  # percentage of features (pixels) to remove

	## random

	for threshold in thresholds:
		if not os.path.exists(f'dataset/roar/random_{threshold}/imgs'):
			print(f'Generating images for threshold {threshold}')
			gen_random(train_loader, threshold, save_path="dataset/roar/", device=device)
			gen_random(val_loader, threshold, save_path="dataset/roar/", device=device)
		else:
			print(f'Images for threshold {threshold} already exist')

	print("Completed random features")

	## LPR

	## SG_SQ

	## concept_removal

	# train models

	# save test accuracies to file
