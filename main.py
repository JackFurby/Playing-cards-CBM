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
from models.cem import XtoCNet as CEM_XtoCNet
from models.cem import End2EndModel as CEM
from dataset.dataset import load_data, find_class_imbalance


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train and test a Concept Bottleneck Model')
	parser.add_argument(
		'--lr',
		type=float,
		default=0.001,
		help='Starting learning rate (default: 0.001)'
	)
	parser.add_argument(
		'--lr_patience',
		type=int,
		default=10,
		help='Number of epochs with no improvement after which learning rate will be reduced (default: 10)'
	)
	parser.add_argument(
		'--mode',
		type=str,
		default="XtoC",
		choices=["XtoC", "standard", "joint", "CtoY", "sequential", "cem", "cem_paper", "cem_XtoC"],
		help='Training method / model to use for training a model (default: XtoC)'
	)
	parser.add_argument(
		'--epochs',
		type=int,
		default=100,
		help='Number of epochs to train a model (default: 100)'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=16,
		help='Number of samples per batch (default: 16)'
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=0,
		help='Seed set to Pytorch (default: 0)'
	)
	parser.add_argument(
		'--num_workers',
		type=int,
		default=0,
		help='Number of workers fetching data for the dataloader (default: 0)'
	)
	parser.add_argument(
		'--optimizer',
		type=str,
		default='sgd',
		choices=["adam", "sgd"],
		help='Optimizer to use for training (default: adam)'
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
		'--weighted_loss',
		action='store_true',
		help='Whether to use weighted loss for single concept or multiple ones'
	)
	parser.add_argument(
		'--attr_loss_weight',
		type=float,
		default=1.0,
		help='Weight task loss (default: 1.0)'
	)
	parser.add_argument(
		'--freeze',
		action='store_true',
		help='Freeze the input to layer before concept vector model part'
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
		'--save_name',
		type=str,
		default="",
		help='Name to save model file. If left as default a systematic file name will be used (default: "")'
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.use_cpu) else "cpu")
	if torch.cuda.is_available() and not args.use_cpu:
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	torch.manual_seed(args.seed)  # reproducibility

	# set name of model file to save
	if args.save_name == "":
		sig_string = "_sig" if args.use_sig else ""
		save_name = f"{args.mode}_{args.seed}{sig_string}.pth"
	else:
		save_name = f"{args.save_name}.pth"

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
		dataset_root=f"{args.dataset_root}/{args.dataset_postfix}",
		attr_loss_weight=args.attr_loss_weight,
		weighted_loss=args.weighted_loss,
		use_sig=args.use_sig,
		freeze=args.freeze,
		seed=args.seed,
		device=device
	)

	wandb.init(project="playing_cards_CBM", config=parameters)
	config = wandb.config

	print(args)  # log args to console

	train_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", args.batch_size, num_concepts=args.n_concepts, dataset_root=args.dataset_root, resampling=True, num_workers=args.num_workers)
	val_loader = load_data(f"./{args.dataset_root}/{args.dataset_postfix}/val.pkl", args.batch_size, num_concepts=args.n_concepts, dataset_root=args.dataset_root, num_workers=args.num_workers)

	if args.mode == "XtoC":
		model = PlayingCardNet(num_concepts=args.n_concepts, pretrained_weights=args.pretrained_weights, freeze=args.freeze)
	elif args.mode == "joint" or args.mode == "sequential":
		model = End2EndModel(num_classes=args.n_classes, num_concepts=args.n_concepts, pretrained_weights=args.pretrained_weights, freeze=args.freeze, use_sigmoid=args.use_sig)
	elif args.mode == "CtoY":
		model = CtoYNet(num_concept_in=args.n_concepts, num_classes=args.n_classes)
	elif args.mode =="cem":
		model = CEM(num_classes=args.n_classes, num_concepts=args.n_concepts, emb_size=16, shared_prob_gen=False, sigmoidal_embedding=False, sigmoidal_prob=True)
		#model = CEM(args.n_concepts, args.n_classes, emb_size=128)
	elif args.mode == "cem_XtoC":
		#model = CEM_XtoCNet(args.n_concepts, emb_size=128)
		model = CEM_XtoCNet(args.n_concepts, emb_size=16, shared_prob_gen=True, sigmoidal_embedding=False, sigmoidal_prob=True)
	elif args.mode =="cem_paper":
		model = ConceptEmbeddingModel(args.n_concepts, args.n_classes, False, emb_size=128)
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

	if args.mode in ["XtoC", "cem_XtoC"]:
		attr_criterion = []  # separate criterion (loss function) for each attribute
		if args.weighted_loss:
			imbalance = find_class_imbalance(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", True)
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
				saveName = f"./saves/{save_name}"
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
				"Train XtoC Loss": trainLoss,
				"Train XtoC Accuracy": trainAcc,
				"Test XtoC Loss": testLoss,
				"Test XtoC Accuracy": testAcc
			}

			if testLoss is not None:
				log_values["trainF1"] = trainF1.item()
				log_values["testF1"] = testF1.item()

			# Log to wandb
			wandb.log(log_values)

			for param_group in optimizer.param_groups:
				wandb.log({"learning rate": param_group['lr']})

	elif args.mode in ["joint", "cem"]:
		attr_criterion = []  # separate criterion (loss function) for each attribute
		attr_criterion.append(nn.CrossEntropyLoss())  # criterion for c->y model out
		if args.weighted_loss:
			imbalance = find_class_imbalance(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", True)
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
				saveName = f"./saves/{save_name}"
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

	elif args.mode in ["standard", "CtoY", "sequential"]:
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
				saveName = f"./saves/{save_name}"
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


	elif args.mode in ["cem_paper"]:
		attr_criterion = []  # separate criterion (loss function) for each attribute
		attr_criterion.append(nn.CrossEntropyLoss())  # criterion for c->y model out
		if args.weighted_loss:
			imbalance = find_class_imbalance(f"./{args.dataset_root}/{args.dataset_postfix}/train.pkl", True)
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
				saveName = f"./saves/{save_name}"
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

	print("Finished training!")
