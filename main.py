import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import wandb
import os

from train import train, test
from models.model import PlayingCardNet
from dataset.dataset import load_data, find_class_imbalance


# Independent & Sequential Model
def ModelXtoC(pretrained, freeze, num_attributes, expand_dim):
	vgg_model = vgg_mini_bn(pretrained=pretrained, num_attributes=num_attributes, bottleneck=True, expand_dim=expand_dim)
	return x_to_c_model(freeze=freeze, model=vgg_model)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train and test a Concept Bottleneck Model')
	parser.add_argument(
		'--lr',
		type=float,
		default=0.001,
		help='Starting learning rate'
	)
	parser.add_argument(
		'--lr_patience',
		type=int,
		default=10,
		help='Number of epochs with no improvement after which learning rate will be reduced'
	)
	parser.add_argument(
		'--mode',
		type=str,
		default="XtoC",
		choices=["XtoC", "standard"],
		help='model to run script'
	)
	parser.add_argument(
		'--epochs',
		type=int,
		default=100,
		help='number of epochs'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=16,
		help='number of epochs'
	)
	parser.add_argument(
		'--num_workers',
		type=int,
		default=0,
		help='number of workers fetching data on the dataloader'
	)
	parser.add_argument(
		'--optimizer',
		type=str,
		default='adam',
		choices=["adam", "sgd"],
		help='Select optimizer to use for training'
	)
	parser.add_argument(
		'--pretrained_weights',
		type=str,
		help='file path to pytorch weight file'
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
		'--dataset_root',
		type=str,
		help='file path to dataset root dir'
	)
	parser.add_argument(
		'--weighted_loss',
		action='store_true',
		help='Whether to use weighted loss for single attribute or multiple ones'
	)
	parser.add_argument(
		'--attr_loss_weight',
		type=float,
		default=1.0,
		help='weight for loss by predicting attributes'
	)
	parser.add_argument(
		'--has_class',
		action='store_true',
		help='Whether the dataset includes class labels'
	)
	parser.add_argument(
		'--freeze',
		action='store_true',
		help='whether to freeze the bottom part of the model'
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	torch.manual_seed(0)  # reproducibility

	parameters = dict(
		batch_size=args.batch_size,
		start_lr=args.lr,
		lr_patience=args.lr_patience,
		epochs=args.epochs,
		training_mode=args.mode,
		pretrained_weights=args.pretrained_weights,
		optimizer=args.optimizer,
		n_concepts=args.n_concepts,
		n_classes=args.n_classes,
		dataset_root=args.dataset_root,
		has_class=args.has_class,
		attr_loss_weight=args.attr_loss_weight,
		weighted_loss=args.weighted_loss,
		freeze=args.freeze
	)

	wandb.init(project="playing_cards_CBM", config=parameters)
	config = wandb.config

	train_loader = load_data(f"./{args.dataset_root}/train.pkl", args.batch_size, dataset_root=args.dataset_root, has_class=args.has_class, resampling=True)
	val_loader = load_data(f"./{args.dataset_root}/val.pkl", args.batch_size, dataset_root=args.dataset_root, has_class=args.has_class)

	if parameters['training_mode'] == "XtoC":
		model = PlayingCardNet(num_concepts=args.n_concepts, pretrained_weights=args.pretrained_weights, freeze=args.freeze)

	else:  # standard training
		model = PlayingCardNet(num_classes=args.n_classes, pretrained_weights=args.pretrained_weights, freeze=args.freeze)

	model.to(device)

	wandb.watch(model, log_freq=100)

	if args.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=0.1)

	if args.mode == "XtoC":
		attr_criterion = []  # separate criterion (loss function) for each attribute
		if args.weighted_loss:
			imbalance = find_class_imbalance(f"./{args.dataset_root}/train.pkl", True)
			print(imbalance)
			for ratio in imbalance:
				attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).to(device)))
		else:
			for i in range(args.n_concepts):
				attr_criterion.append(torch.nn.CrossEntropyLoss())

		bestAccuracy = None

		for epoch in range(parameters['epochs']):
			trainLoss, trainAcc = train(model, train_loader, optimizer, attr_criterion, mode=args.mode, attr_loss_weight=args.attr_loss_weight, device=device)
			testLoss, testAcc = test(model, val_loader, optimizer, attr_criterion, mode=args.mode, attr_loss_weight=args.attr_loss_weight, device=device)

			scheduler.step(testLoss)

			if (bestAccuracy is None) or (bestAccuracy < testAcc):
				bestAccuracy = testAcc

				if not os.path.exists("./saves"):
					os.makedirs("./saves")
				saveName = "./saves/XtoC_best.pth"
				torch.save(model.state_dict(), saveName)
				wandb.save(saveName)
				print("Model saved")

			print("epoch:", epoch)
			print("trainLoss:", trainLoss)
			print("trainAcc:", trainAcc, "%")
			print("testLoss:", testLoss)
			print("testAcc:", testAcc, "%")

			# Log to wandb
			wandb.log({
				"epoch": epoch,
				"XtoC Train Loss": trainLoss,
				"XtoC Train Accuracy": trainAcc,
				"XtoC Test Loss": testLoss,
				"XtoC Test Accuracy": testAcc
			})
			for param_group in optimizer.param_groups:
				wandb.log({"XtoC learning rate": param_group['lr']})

	elif args.mode == "standard":
		criterion = nn.CrossEntropyLoss()

		bestAccuracy = None

		for epoch in range(parameters['epochs']):
			trainLoss, trainAcc = train(model, train_loader, optimizer, criterion, mode=args.mode, device=device)
			testLoss, testAcc = test(model, val_loader, optimizer, criterion, mode=args.mode, device=device)

			scheduler.step(testLoss)

			if (bestAccuracy is None) or (bestAccuracy < testAcc):
				bestAccuracy = testAcc

				if not os.path.exists("./saves"):
					os.makedirs("./saves")
				saveName = "./saves/playing_card_best.pth"
				torch.save(model.state_dict(), saveName)
				wandb.save(saveName)
				print("Model saved")

			print("epoch:", epoch)
			print("trainLoss:", trainLoss)
			print("trainAcc:", 100 * trainAcc, "%")
			print("testLoss:", testLoss)
			print("testAcc:", 100 * testAcc, "%")

			# Log to wandb
			wandb.log({
				"epoch": epoch,
				"XtoC Train Loss": trainLoss,
				"XtoC Train Accuracy": 100 * trainAcc,
				"XtoC Test Loss": testLoss,
				"XtoC Test Accuracy": 100 * testAcc
			})
			for param_group in optimizer.param_groups:
				wandb.log({"XtoC learning rate": param_group['lr']})

	print("Finished training!")
