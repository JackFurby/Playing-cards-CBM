import torch
from torch import nn


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/template_model.py
class End2EndModel(torch.nn.Module):
	def __init__(self, num_classes=6, num_concepts=52, num_images_per_output=3, pretrained_weights=None, freeze=False, use_sigmoid=False):
		super(End2EndModel, self).__init__()
		self.x_to_c_model = PlayingCardNet(num_concepts=args.n_concepts, pretrained_weights=args.pretrained_weights, freeze=args.freeze)
		self.c_to_y_model = nn.Linear(num_concepts*num_images_per_output, num_classes)

	def forward_stage2(self, image_1_out, image_2_out, image_3_out):
		if self.use_sigmoid:
			outputs = torch.nn.Sigmoid()(torch.cat((torch.cat(image_1_out, dim=1), torch.cat(image_2_out, dim=1),torch.cat(image_3_out, dim=1)), dim=0))
		else:
			outputs = torch.cat((image_1_out, image_2_out, image_3_out), dim=0)

		return (outputs, self.c_to_y_model(outputs))

	def forward(self, x):
		output_image_1 = self.x_to_c_model(x[0])
		output_image_2 = self.x_to_c_model(x[1])
		output_image_3 = self.x_to_c_model(x[2])
		return self.forward_stage2(output_image_1, output_image_2, output_image_3)


class PlayingCardNet(nn.Module):
	def __init__(self, num_classes=52, num_concepts=0, pretrained_weights=None, freeze=False):
		super(PlayingCardNet, self).__init__()
		self.num_concepts = num_concepts
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, 3),  # 3 channels; 64 is the output channel size; 3 is the kernel size;
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 64, 3),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 128, 3),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(128, 128, 3),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.avgPool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.5),
			nn.Linear(6272, 512),
			nn.ReLU()
		)
		# concept training requires a module list for loss
		if self.num_concepts > 0:
			# all_fc replaces fc2 from standard model
			self.all_fc = nn.ModuleList()  #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
			for i in range(self.num_concepts):
				self.all_fc.append(nn.Linear(512, 1))
		# basic fc for standdard training
		else:
			self.out_fc = nn.Linear(512, num_classes)

		# if weights file provided, load them
		if pretrained_weights is not None:
			self.load_state_dict(torch.load(pretrained_weights, map_location=torch.device('cpu')), strict=False)  # as we are replacing the final fc layer we can set strict to False
			print(f"Loaded weights from {pretrained_weights}")

		# if freeze == true then freeze all but fc layers
		if freeze:  # only finetune fc layer
			for name, param in self.named_parameters():
				if 'fc' not in name:
					param.requires_grad = False
			print("frozen lower model layers")

	def forward(self, x):
		x = self.features(x)
		x = self.avgPool(x)
		x = self.classifier(x)
		if self.num_concepts > 0:
			out = []
			for fc in self.all_fc:
				out.append(fc(x))
		else:
			out = self.out_fc(x)
		return out
