import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchmetrics import F1Score
import argparse
import wandb
import os

from train import train
from models.model import PlayingCardNet, End2EndModel, CtoYNet
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
		choices=["XtoC", "standard", "joint", "CtoY"],
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
		'--num_images_per_output',
		type=int,
		default=1,
		help='number of images input for each output (only used for joint training)'
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
		num_images_per_output=args.num_images_per_output,
		dataset_root=args.dataset_root,
		attr_loss_weight=args.attr_loss_weight,
		weighted_loss=args.weighted_loss,
		freeze=args.freeze
	)

	wandb.init(project="playing_cards_CBM", config=parameters)
	config = wandb.config

	if args.mode in ["joint", "CtoY"] and args.num_images_per_output != 1:
		train_loader = load_data(f"./{args.dataset_root}/train.pkl", args.batch_size, num_concepts=args.n_concepts, dataset_root=args.dataset_root, num_workers=args.num_workers, multi_image=True)
		val_loader = load_data(f"./{args.dataset_root}/val.pkl", args.batch_size, num_concepts=args.n_concepts, dataset_root=args.dataset_root, num_workers=args.num_workers, multi_image=True)
	else:
		train_loader = load_data(f"./{args.dataset_root}/train.pkl", args.batch_size, num_concepts=args.n_concepts, dataset_root=args.dataset_root, resampling=True, num_workers=args.num_workers)
		val_loader = load_data(f"./{args.dataset_root}/val.pkl", args.batch_size, num_concepts=args.n_concepts, dataset_root=args.dataset_root, num_workers=args.num_workers)

	if args.mode == "XtoC":
		model = PlayingCardNet(num_concepts=args.n_concepts, pretrained_weights=args.pretrained_weights, freeze=args.freeze)
	elif args.mode == "joint":
		model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, num_images_per_output=args.num_images_per_output, pretrained_weights=args.pretrained_weights, freeze=args.freeze, use_sigmoid=True)
	elif args.mode == "CtoY":
		model = CtoYNet(num_concept_in=args.n_concepts*args.num_images_per_output, num_classes=args.n_classes)
	else:  # standard training
		model = PlayingCardNet(num_classes=args.n_classes, pretrained_weights=args.pretrained_weights, freeze=args.freeze)

	model.to(device)

	wandb.watch(model, log_freq=100)

	if args.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=0.1)

	f1 = F1Score().to(device)

	if args.mode == "XtoC":
		attr_criterion = []  # separate criterion (loss function) for each attribute
		if args.weighted_loss:
			imbalance = find_class_imbalance(f"./{args.dataset_root}/train.pkl", True)
			print(imbalance)
			for ratio in imbalance:
				attr_criterion.append(nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).to(device)))
		else:
			for i in range(args.n_concepts):
				attr_criterion.append(nn.CrossEntropyLoss())

		bestAccuracy = None

		for epoch in range(parameters['epochs']):
			trainLoss, trainAcc, trainF1 = train(model, train_loader, optimizer, attr_criterion, f1=f1, mode=args.mode, attr_loss_weight=args.attr_loss_weight, train=True, device=device)
			testLoss, testAcc, testF1 = train(model, val_loader, optimizer, attr_criterion, f1=f1, mode=args.mode, attr_loss_weight=args.attr_loss_weight, train=False, device=device)

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
			print("trainF1:", trainF1.item())
			print("testLoss:", testLoss)
			print("testAcc:", testAcc, "%")
			print("testF1:", testF1.item())

			log_values = {
				"epoch": epoch,
				"XtoC Train Loss": trainLoss,
				"XtoC Train Accuracy": trainAcc,
				"XtoC Test Loss": testLoss,
				"XtoC Test Accuracy": testAcc
			}

			if testLoss is not None:
				log_values["trainF1"] = trainF1.item()
				log_values["testF1"] = testF1.item()

			# Log to wandb
			wandb.log(log_values)

			for param_group in optimizer.param_groups:
				wandb.log({"learning rate": param_group['lr']})

	elif args.mode == "joint":
		attr_criterion = []  # separate criterion (loss function) for each attribute
		attr_criterion.append(nn.CrossEntropyLoss())  # criterion for c->y model out
		if args.weighted_loss:
			imbalance = find_class_imbalance(f"./{args.dataset_root}/train.pkl", True)
			print(imbalance)
			for ratio in imbalance:
				attr_criterion.append(nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).to(device)))
		else:
			for i in range(args.n_concepts):
				attr_criterion.append(nn.CrossEntropyLoss())

		bestAccuracy = None

		for epoch in range(parameters['epochs']):
			trainLoss, trainAcc, trainF1, trainTaskF1, trainTaskAcc = train(model, train_loader, optimizer, attr_criterion, f1=f1, mode=args.mode, attr_loss_weight=args.attr_loss_weight, train=True, device=device)
			testLoss, testAcc, testF1, testTaskF1, testTaskAcc = train(model, val_loader, optimizer, attr_criterion, f1=f1, mode=args.mode, attr_loss_weight=args.attr_loss_weight, train=False, device=device)

			scheduler.step(testLoss)

			if (bestAccuracy is None) or (bestAccuracy < testAcc):
				bestAccuracy = testAcc

				if not os.path.exists("./saves"):
					os.makedirs("./saves")
				saveName = "./saves/XtoCtoY_best.pth"
				torch.save(model.state_dict(), saveName)
				wandb.save(saveName)
				print("Model saved")

			print("epoch:", epoch)
			print("trainLoss:", trainLoss)
			print("trainAcc:", trainAcc, "%")
			print("trainTaskAcc:", 100 * trainTaskAcc, "%")
			print("trainF1:", trainF1.item())
			print("trainTaskF1:", trainTaskF1.item())
			print("testLoss:", testLoss)
			print("testAcc:", testAcc, "%")
			print("testTaskAcc:", 100 * testTaskAcc, "%")
			print("testF1:", testF1.item())
			print("testTaskF1:", testTaskF1.item())

			log_values = {
				"epoch": epoch,
				"Train Loss": trainLoss,
				"Train XtoC Accuracy": trainAcc,
				"Train Task Accuracy": 100 * trainTaskAcc,
				"Test Loss": testLoss,
				"Test XtoC Accuracy": testAcc,
				"Test Task Accuracy": 100 * testTaskAcc
			}

			if testLoss is not None:
				log_values["trainF1"] = trainF1.item()
				log_values["testF1"] = testF1.item()
				log_values["trainTaskF1"] = trainTaskF1.item()
				log_values["testTaskF1"] = testTaskF1.item()

			# Log to wandb
			wandb.log(log_values)

			for param_group in optimizer.param_groups:
				wandb.log({"learning rate": param_group['lr']})

	elif args.mode in ["standard", "CtoY"]:
		criterion = nn.CrossEntropyLoss()

		bestAccuracy = None

		for epoch in range(parameters['epochs']):
			trainLoss, trainAcc, trainF1 = train(model, train_loader, optimizer, criterion, f1=f1, mode=args.mode, train=True, device=device)
			testLoss, testAcc, testF1 = train(model, val_loader, optimizer, criterion, f1=f1, mode=args.mode, train=False, device=device)

			scheduler.step(testLoss)

			if (bestAccuracy is None) or (bestAccuracy < testAcc):
				bestAccuracy = testAcc

				if not os.path.exists("./saves"):
					os.makedirs("./saves")
				saveName = f"./saves/{args.mode}_playing_card_best.pth"
				torch.save(model.state_dict(), saveName)
				wandb.save(saveName)
				print("Model saved")

			print("epoch:", epoch)
			print("trainLoss:", trainLoss)
			print("trainAcc:", 100 * trainAcc, "%")
			print("trainF1:", trainF1.item())
			print("testLoss:", testLoss)
			print("testAcc:", 100 * testAcc, "%")
			print("testF1:", testF1.item())

			log_values = {
				"epoch": epoch,
				"Train Loss": trainLoss,
				"Train Task Accuracy": 100 * trainAcc,
				"Test Loss": testLoss,
				"Test Task Accuracy": 100 * testAcc
			}

			if testLoss is not None:
				log_values["trainF1"] = trainF1.item()
				log_values["testF1"] = testF1.item()

			# Log to wandb
			wandb.log(log_values)

			for param_group in optimizer.param_groups:
				wandb.log({"learning rate": param_group['lr']})

	print("Finished training!")
