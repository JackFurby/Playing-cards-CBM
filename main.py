import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import wandb
import os

from train import train, test
from models.model import PlayingCardNet, ConcpetPlayingCardNet
from dataset.dataset import load_data, find_class_imbalance


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
		'--pretrained',
		action='store_true',
		help='Sets whether the model loads pretrained weights'
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
		'--use_relu',
		action='store_true',
		help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model'
	)
	parser.add_argument(
		'--use_sigmoid',
		action='store_true',
		help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model'
	)
	parser.add_argument(
		'--weighted_loss',
		action='store_true',
		help='Whether to use weighted loss for single attribute or multiple ones'
	)
	parser.add_argument(
		'--attr_loss_weight',
		type=float,
		default=0.01,
		help='weight for loss by predicting attributes'
	)
	parser.add_argument(
		'--has_class',
		action='store_true',
		help='Whether the dataset includes class labels'
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
		pretrained=args.pretrained,
		optimizer=args.optimizer,
		n_concepts=args.n_concepts,
		n_classes=args.n_classes,
		use_relu=args.use_relu,
		use_sigmoid=args.use_sigmoid,
		dataset_root=args.dataset_root,
		has_class=args.has_class,
		attr_loss_weight=args.attr_loss_weight,
		weighted_loss=args.weighted_loss
	)

	wandb.init(project="playing_cards_CBM", config=parameters)
	config = wandb.config

	train_loader = load_data(f"./{args.dataset_root}/train.pkl", args.batch_size, dataset_root=args.dataset_root, has_class=args.has_class)
	val_loader = load_data(f"./{args.dataset_root}/val.pkl", args.batch_size, dataset_root=args.dataset_root, has_class=args.has_class)

	if parameters['training_mode'] == "XtoC":
		model = ConcpetPlayingCardNet(num_concepts=args.n_concepts, pretrained=args.pretrained, pretrained_weights=args.pretrained_weights)
		#model = PlayingCardNet(num_classes=args.n_concepts, pretrained=args.pretrained, pretrained_weights=args.pretrained_weights)
	else:  # standard training
		model = PlayingCardNet(num_classes=args.n_classes, pretrained=args.pretrained, pretrained_weights=args.pretrained_weights)

	model.to(device)

	wandb.watch(model, log_freq=100)

	if args.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=parameters['start_lr'])
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr=parameters['start_lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=parameters['lr_patience'], factor=0.1)

	if args.mode == "XtoC":
		attr_criterion = []  # separate criterion (loss function) for each attribute
		if args.weighted_loss:
			imbalance = find_class_imbalance(f"./{args.dataset_root}/train.pkl", True)
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

		print("Finished training!")

	elif args.mode == "standard":
		criterion = nn.BCEWithLogitsLoss()

		bestAccuracy = None

		for epoch in range(parameters['epochs']):
			trainLoss, trainAcc = train(model, train_loader, optimizer, criterion, mode=args.mode, device=device)
			testLoss, testAcc = test(model, val_loader, optimizer, criterion, mode=args.mode, device=device)

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

		print("Finished training!")
