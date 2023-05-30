# Modified from https://github.com/jfhealthcare/Chexpert/blob/master/model/utils.py
import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop


def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
	if norm_type == 'BatchNorm':
		return nn.BatchNorm2d(num_features, eps=eps)
	elif norm_type == "GroupNorm":
		return nn.GroupNorm(num_groups, num_features, eps=eps)
	elif norm_type == "InstanceNorm":
		return nn.InstanceNorm2d(num_features, eps=eps, affine=True, track_running_stats=True)
	else:
		raise Exception('Unknown Norm Function : {}'.format(norm_type))
