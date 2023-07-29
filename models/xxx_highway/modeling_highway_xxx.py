from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from modeling_deit import DeiTEmbeddings, DeiTLayer, DeiTPreTrainedModel, DeiTPatchEmbeddings
from configuration_deit import DeiTConfig


def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    x = torch.softmax(x, dim=-1)               # softmax normalized prob distribution
    return -torch.sum(x*torch.log(x), dim=-1)  # entropy calculation on probs: -\sum(p \ln(p))


class xxxEncoder(nn.Module):
class xxxPooler(nn.Module):
class xxxModel(xxxPreTrainedModel):
class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start form 1!

class xxxHighwayExits(nn.Module):
    r'''
    A module to provide a shortcut from 
    the output of one non-final DeiTLayer in DeiTEncoder to 
    cross-entropy computation in DeiTForImageClassification
    '''
class xxxHighwayForImageClassification(xxxPreTrainedModel):
