import torch
from torch import nn


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
		self.c_to_y_model = CtoYNet(num_concepts*emb_size, num_classes)
		self.x_to_c_model = XtoCNet(num_concepts, emb_size, shared_prob_gen, sigmoidal_embedding, sigmoidal_prob)

		# if freeze == true then freeze all layers in x_to_c_model
		if freeze:  # only finetune c_to_y_model
			for name, param in self.x_to_c_model.named_parameters():
				param.requires_grad = False
			print("frozen layers in x_to_c_model")

	def forward_stage2(self, stage_1_out):
		c_to_y_input = stage_1_out[1]

		return (stage_1_out[0], self.c_to_y_model(c_to_y_input))

	def forward(self, x, train=False):
		return self.forward_stage2(self.x_to_c_model(x, train=train))


class XtoCNet(nn.Module):
	def __init__(self, num_concepts, emb_size, shared_prob_gen, sigmoidal_embedding, sigmoidal_prob, pretrained_weights=None, concat_prob=False):
		super(XtoCNet, self).__init__()
		self.pre_concept_model = PlayingCardNet(out_size=num_concepts, pretrained_weights=pretrained_weights)

		self.concept_context_generators = torch.nn.ModuleList()
		self.shared_prob_gen = shared_prob_gen
		self.sigmoidal_embedding = sigmoidal_embedding
		self.sigmoidal_prob = sigmoidal_prob
		self.emb_size = emb_size
		self.concat_prob = concat_prob
		self.training_intervention_prob = 0.25
		self.ones = torch.ones(num_concepts)

		for i in range(num_concepts):
			self.concept_context_generators.append(
				torch.nn.Sequential(*[
					torch.nn.Linear(num_concepts, 2 * emb_size),  # positive and negative embedding parts
					torch.nn.LeakyReLU(),
				])
			)

		if self.shared_prob_gen:
			self.concept_prob_generators = torch.nn.Linear(2 * emb_size, 1)
		else:
			self.concept_prob_generators = torch.nn.ModuleList()
			for i in range(num_concepts):
				self.concept_prob_generators.append(
					torch.nn.Linear(2 * emb_size, 1)
				)

	def _after_interventions(self, prob, concept_idx, intervention_idxs=None, c_true=None, train=False,):
		if train and (self.training_intervention_prob != 0) and (
			intervention_idxs is None
		):
			# Then we will probabilistically intervene in some concepts
			mask = torch.bernoulli(self.ones * self.training_intervention_prob)
			intervention_idxs = torch.nonzero(mask).reshape(-1)
		if (c_true is None) or (intervention_idxs is None):
			return prob
		if concept_idx not in intervention_idxs:
			return prob
		c_true = self._switch_concepts(c_true)
		if self.sigmoidal_prob:
			return c_true[:, concept_idx:concept_idx+1]
		result = (
			(
				c_true[:, concept_idx:concept_idx+1] *
				self.active_intervention_values[concept_idx]
			) +
			(
				(c_true[:, concept_idx:concept_idx+1] - 1) *
				-self.inactive_intervention_values[concept_idx]
			)
		)
		return result

	def forward(self, x, intervention_idxs=None, c=None, train=False):
		h = self.pre_concept_model(x)
		probs = []
		full_vectors = []
		sem_probs = []
		for i, context_gen in enumerate(self.concept_context_generators):
			if self.shared_prob_gen:
				prob_gen = self.concept_prob_generators
			else:
				prob_gen = self.concept_prob_generators[i]

			context = context_gen(h)
			if self.sigmoidal_embedding:
				context = torch.nn.Sigmoid()(context)

			prob = prob_gen(context)
			sem_probs.append(torch.nn.Sigmoid()(prob))
			if self.sigmoidal_prob:
				prob = torch.nn.Sigmoid()(prob)
			prob = self._after_interventions(prob, concept_idx=i, intervention_idxs=intervention_idxs, c_true=c, train=train)
			probs.append(prob)

			context_pos = context[:, :self.emb_size]
			context_neg = context[:, self.emb_size:]
			mask = prob if self.sigmoidal_prob else torch.nn.Sigmoid()(prob)

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

		c_sem = torch.cat(sem_probs, axis=-1)
		c_pred = torch.cat(full_vectors, axis=-1)
		return c_sem, c_pred


class CtoYNet(nn.Module):
	def __init__(self, num_concept_in, num_classes):
		super(CtoYNet, self).__init__()
		self.classifier = nn.Sequential(
			nn.Linear(num_concept_in, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes)
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
