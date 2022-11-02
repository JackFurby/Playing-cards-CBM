import torch
from torch import nn
from models.cbm2 import ConceptBottleneckModel

from torchvision.models import vgg11_bn


class End2EndModel(torch.nn.Module):
	def __init__(
		self,
		num_classes=6,
		num_concepts=52,
		emb_size=16,
		pretrained_weights=None,
		freeze=False,
		concat_prob=False
	):
		super(End2EndModel, self).__init__()
		self.x_to_c_model = XtoCNet(num_concepts, num_classes, emb_size=emb_size)
		self.c_to_y_model = CtoYNet(num_concepts * (emb_size + int(concat_prob)), num_classes)

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


class XtoCNet(torch.nn.Module):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        pretrain_model=True,
        c_extractor_arch=vgg11_bn,
        shared_prob_gen=True, #False,
        intervention_idxs=None,
        sigmoidal_prob=True, #False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.25, #0.0,
        active_intervention_values=None,
        inactive_intervention_values=None,
        embeding_activation="leakyrelu", # None
        concat_prob=False
    ):
        super(XtoCNet, self).__init__()
        try:
            self.pre_concept_model = c_extractor_arch(
                pretrained=pretrain_model
            )
        except:
            self.pre_concept_model = c_extractor_arch(num_classes=n_concepts)

        #self.pre_concept_model = PlayingCardNet(out_size=n_concepts)
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts) * (
                5.0 if not sigmoidal_prob else 1.0
            )
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts) * (
                -5.0 if not sigmoidal_prob else 0.0
            )
        self.concat_prob = concat_prob
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.intervention_idxs = intervention_idxs
        for i in range(n_concepts):
            if embeding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        # Two as each concept will have a positive and a
                        # negative embedding portion which are later mixed
                        2 * emb_size,
                    )
                )
            elif embeding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embeding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.ReLU(),
                    ])
                )
            if self.shared_prob_gen and (
                len(self.concept_prob_generators) == 0
            ):
                # Then we will use one and only one probability generator which
                # will be shared among all concepts. This will force concept
                # embedding vectors to be pushed into the same latent space
                self.concept_prob_generators.append(
                    torch.nn.Linear(
                        2 * emb_size,
                        1,
                    )
                )
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(
                    torch.nn.Linear(
                        2 * emb_size,
                        1,
                    )
                )
        self.sig = torch.nn.Sigmoid()

        self.loss_task = (
            torch.nn.CrossEntropyLoss()
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss()
        )
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_embedding = sigmoidal_embedding
        self.emb_size = emb_size

    def _after_interventions(
        self,
        prob,
        concept_idx,
        intervention_idxs=None,
        c_true=None,
        train=False,
    ):
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
        pre_c = self.pre_concept_model(x)
        probs = []
        full_vectors = []
        sem_probs = []
        for i, context_gen in enumerate(self.concept_context_generators):
            if self.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[i]
            context = context_gen(pre_c)
            if self.sigmoidal_embedding:
                context = self.sig(context)

            prob = prob_gen(context)
            sem_probs.append(self.sig(prob))
            if self.sigmoidal_prob:
                prob = self.sig(prob)
            prob = self._after_interventions(
                prob,
                concept_idx=i,
                intervention_idxs=intervention_idxs,
                c_true=c,
                train=train,
            )
            probs.append(prob)
            # Then time to mix!
            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            mask = prob if self.sigmoidal_prob else self.sig(prob)
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
class PlayingCardNet(torch.nn.Module):
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
