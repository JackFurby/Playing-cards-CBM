import torch
from torch import nn

from torchvision.models import vgg11_bn


class End2EndModel(torch.nn.Module):
	def __init__(
		self,
		num_classes=6,
		num_concepts=52,
		emb_size=16,
		pretrained_weights=None,
		freeze=False,
		sigmoidal_embedding=False,
		sigmoidal_prob=False,
		shared_prob_gen=True
	):
		super(End2EndModel, self).__init__()
		self.x_to_c_model = XtoCNet(num_concepts, emb_size, shared_prob_gen, sigmoidal_embedding, sigmoidal_prob)
		self.c_to_y_model = CtoYNet(num_concepts*emb_size, num_classes)

		# if freeze == true then freeze all layers in x_to_c_model
		if freeze:  # only finetune c_to_y_model
			for name, param in self.x_to_c_model.named_parameters():
				param.requires_grad = False
			print("frozen layers in x_to_c_model")

	def forward_stage2(self, stage_1_out):
		c_sem, c_pred = stage_1_out
		y = self.c_to_y_model(c_pred)

		return c_sem, y

	def forward(self, x, train=False):
		return self.forward_stage2(self.x_to_c_model(x, train=train))


class XtoCNet(nn.Module):
	def __init__(self, num_concepts, emb_size, shared_prob_gen, sigmoidal_embedding, sigmoidal_prob, pretrained_weights=None, concat_prob=False, training_intervention_prob=0.25):
		super(XtoCNet, self).__init__()
		self.pre_concept_model = PlayingCardNet(out_size=num_concepts, pretrained_weights=pretrained_weights)
		pre_c_out_size = list(self.pre_concept_model.modules())[-1].out_features

		self.concept_context_generators = torch.nn.ModuleList()
		self.shared_prob_gen = shared_prob_gen
		self.sigmoidal_embedding = sigmoidal_embedding
		self.sigmoidal_prob = sigmoidal_prob
		self.emb_size = emb_size
		self.concat_prob = concat_prob
		self.training_intervention_prob=training_intervention_prob
		if self.training_intervention_prob != 0:
			self.ones = torch.ones(num_concepts)

		for i in range(num_concepts):
			self.concept_context_generators.append(
				torch.nn.Sequential(
					torch.nn.Linear(pre_c_out_size, 2 * emb_size),  # positive and negative embedding parts
					torch.nn.ReLU(),
				)
			)

		if self.shared_prob_gen:
			self.concept_prob_generators = torch.nn.Linear(2 * emb_size, 1)
		else:
			self.concept_prob_generators = torch.nn.ModuleList()
			for i in range(num_concepts):
				self.concept_prob_generators.append(
					torch.nn.Linear(2 * emb_size, 1)
				)

	def _after_interventions(self, prob, train):
		if train and (self.training_intervention_prob != 0):
			# Then we will probabilistically set some concepts to 0
			ones = torch.ones(prob.size())
			mask = torch.bernoulli(ones * self.training_intervention_prob)
			prob = prob * mask
		return prob

	def forward(self, x, train=False, return_type="all"):
		h = self.pre_concept_model(x)
		probs = []  # concept probabilitie predictions, may be modified with interventions
		full_vectors = []  # bottleneck vectors
		sem_probs = []  # true concept probability predictions

		# for each concept predict the probibility and concept vector (using probability and context)
		for i, context_gen in enumerate(self.concept_context_generators):
			if self.shared_prob_gen:
				prob_gen = self.concept_prob_generators
			else:
				prob_gen = self.concept_prob_generators[i]

			context = context_gen(h)
			if self.sigmoidal_embedding:
				context = torch.nn.Sigmoid()(context)

			prob = prob_gen(context)
			sem_probs.append(prob)

			prob = self._after_interventions(prob, train=train)
			probs.append(prob)

			context_pos = context[:, :self.emb_size]
			context_neg = context[:, self.emb_size:]

			if self.sigmoidal_prob:
				mask = torch.nn.Sigmoid()(prob)
			else:
				mask = prob


			context = context_pos * mask + context_neg * (1 - mask)
			if self.concat_prob:
				# Then the probability bit will be added
				# as part of the bottleneck
				full_vectors.append(torch.cat(
					[context, probs[i]],
					axis=-1,
				))
			else:
				# Otherwise, let's completely ignore the probability bit
				full_vectors.append(context)

		c_sem = sem_probs  # concept probability predictions
		c_pred = torch.cat(full_vectors, axis=-1)  # bottleneck vectors

		if return_type == "probs":
			return torch.cat(c_sem, dim=1)
		if return_type == "concepts":
			return c_pred
		else:
			return c_sem, c_pred


class CtoYNet(nn.Module):
	def __init__(self, num_concept_in, num_classes):
		super(CtoYNet, self).__init__()
		self.classifier = nn.Sequential(
			nn.Linear(num_concept_in, 512),
			nn.ReLU(),
			nn.Linear(512, num_classes)
		)

	def forward(self, x):
		return self.classifier(x)


# VGG 11 with Batch Norm
class PlayingCardNet(nn.Module):
	def __init__(self, out_size=52, pretrained_weights=None, freeze=False):
		super(PlayingCardNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.avgPool = nn.AdaptiveAvgPool2d((7, 7))
		self.flatten = nn.Flatten()
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
		)
		self.out_fc = nn.Linear(4096, out_size)

		# if weights file provided, load them
		if pretrained_weights is not None:
			self.load_state_dict(torch.load(pretrained_weights, map_location=torch.device('cpu')), strict=False)  # as we are replacing the final fc layer we can set strict to False
			print(f"Loaded weights from {pretrained_weights}")

		# if freeze == true then freeze all but classifier layers
		if freeze:
			for name, param in self.features.named_parameters():
				param.requires_grad = False
			print("frozen none model layers")

	def forward(self, x):
		x = self.features(x)
		x = self.avgPool(x)
		x = self.flatten(x)
		x = self.classifier(x)
		return self.out_fc(x)
