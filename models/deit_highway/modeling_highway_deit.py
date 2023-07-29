from dataclasses import dataclass
from collections import Iterable
from typing import Optional, Set, Tuple, Union

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PretrainedConfig
from modeling_deit import DeiTEmbeddings, DeiTLayer, DeiTPreTrainedModel, DeiTPatchEmbeddings
from configuration_deit import DeiTConfig
from models.deit_highway.highway import ViTHighway, DeiTHighway, DeiTHighway_v2, ViT_EE_Highway


def CrossEntropy(outputs, targets, temperature):
    log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
    softmax_targets = F.softmax(targets / temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    x = torch.softmax(x, dim=-1)  # softmax normalized prob distribution
    return -torch.sum(x * torch.log(x), dim=-1)  # entropy calculation on probs: -\sum(p \ln(p))


def confidence(x):
    # x: torch.Tensor, logits BEFORE softmax
    softmax = torch.softmax(x, dim=-1)
    return torch.max(softmax)


def prediction(x):
    # x: torch.Tensor, logits BEFORE softmax
    softmax = torch.softmax(x, dim=-1)
    return torch.argmax(softmax)

class DeiTEncoder(nn.Module):
    def __init__(self, config: DeiTConfig):
        super(DeiTEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([DeiTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.num_early_exits = config.num_early_exits
        
        print(f'backbone:{config.backbone}')
        print(f'exit_type:{config.highway_type}')
        self.init_highway()

        self.exit_strategy = config.exit_strategy
        self.train_strategy = config.train_strategy
        
        self.use_lte = True if self.exit_strategy == 'gumbel_lte' else False
        print(f'use_lte:{self.use_lte}')
        
        self.set_early_exit_threshold(self.config.threshold)
        self.set_early_exit_position()


    def init_highway(self):
        config = self.config
        if config.highway_type == 'linear':
            if config.backbone == 'ViT':
                self.highway = nn.ModuleList([ViTHighway(config) for _ in range(config.num_early_exits)])
            elif config.backbone == 'DeiT':
                self.highway = nn.ModuleList([DeiTHighway(config) for _ in range(config.num_early_exits)])
        elif config.highway_type == 'LGViT' and config.num_early_exits == 8:
            self.highway = nn.ModuleList([
                DeiTHighway_v2(config, highway_type=f'conv1_1'),
                DeiTHighway_v2(config, highway_type=f'conv1_1'),
                DeiTHighway_v2(config, highway_type=f'conv2_1'),
                DeiTHighway_v2(config, highway_type=f'conv2_1'),
                DeiTHighway_v2(config, highway_type=f'attention_r2'),
                DeiTHighway_v2(config, highway_type=f'attention_r2'),
                DeiTHighway_v2(config, highway_type=f'attention_r3'),
                DeiTHighway_v2(config, highway_type=f'attention_r3'),
            ])
        elif config.highway_type == 'LGViT' and config.num_early_exits == 2:
            self.highway = nn.ModuleList([
                DeiTHighway_v2(config, highway_type=f'conv1_1'),
                DeiTHighway_v2(config, highway_type=f'attention_r2'),
            ])
        elif config.highway_type == 'LGViT' and config.num_early_exits == 4:
            self.highway = nn.ModuleList([
                DeiTHighway_v2(config, highway_type=f'conv1_1'),
                DeiTHighway_v2(config, highway_type=f'conv2_1'),
                DeiTHighway_v2(config, highway_type=f'attention_r2'),
                DeiTHighway_v2(config, highway_type=f'attention_r3'),
            ])
        elif config.highway_type == 'LGViT' and config.num_early_exits == 6:
            self.highway = nn.ModuleList([
                DeiTHighway_v2(config, highway_type=f'conv1_1'),
                DeiTHighway_v2(config, highway_type=f'conv2_1'),
                DeiTHighway_v2(config, highway_type=f'attention_r2'),
                DeiTHighway_v2(config, highway_type=f'attention_r2'),
                DeiTHighway_v2(config, highway_type=f'attention_r3'),
                DeiTHighway_v2(config, highway_type=f'attention_r3'),
            ])
        elif config.highway_type == 'vit':
            self.highway = nn.ModuleList([ViT_EE_Highway(config) for _ in range(config.num_early_exits)])
        else:
            self.highway = nn.ModuleList([DeiTHighway_v2(config,config.highway_type) for _ in range(config.num_early_exits)])
            
            
    def set_early_exit_position(self):

        num_hidden_layers = self.config.num_hidden_layers
        self.num_early_exits = self.config.num_early_exits
        position_exits = self.config.position_exits

        self.position_exits = [6, 7, 8, 9, 10, 11, 12]

        if position_exits is not None and isinstance(position_exits, Iterable):
            position_exits = eval(self.config.position_exits)
            if len(position_exits) != self.num_early_exits:
                raise ValueError(
                    "Lengths of config.position_exits and num_early_exits do not match, which can lead to poor training results!")
            else:
                self.position_exits = position_exits
        # self.position_exits = [i for i in range(self.num_early_exits)]

        print('The exits are in position:', position_exits)
        self.position_exits = {int(position) - 1: index for index, position in enumerate(self.position_exits)}

        # self.position_exits = {int((num_hidden_layers/self.num_early_exits)*(i+1))-1:i for i in range(self.num_early_exits)}

    def set_early_exit_threshold(self, x=None):

        if self.exit_strategy == 'entropy':
            self.early_exit_threshold = [0.6 for _ in range(self.config.num_early_exits)]
        elif self.exit_strategy == 'confidence':
            self.early_exit_threshold = [0.865 for _ in range(self.config.num_early_exits)]
        elif self.exit_strategy == 'patience':
            self.early_exit_threshold = (7,)
        elif self.exit_strategy == 'patient_and_confident':
            self.early_exit_threshold = [0.5 for _ in range(self.config.num_early_exits)]
            self.early_exit_threshold.append(2)

        if x is not None:
            if (type(x) is float) or (type(x) is int):
                for i in range(len(self.early_exit_threshold)):
                    self.early_exit_threshold[i] = x
            else:
                self.early_exit_threshold = x

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
    ) -> tuple:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_highway_exits = ()
                
        if self.exit_strategy == 'patience':
            # store the number of times that the predictions remain “unchanged”
            cnt = 0
        elif self.exit_strategy == 'patient_and_confident':
            # store the number of times that the predictions remain confident in consecutive layers
            pct = 0

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            current_outputs = (hidden_states,)
            if output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if output_attentions:
                current_outputs = current_outputs + (all_self_attentions,)

            if i in self.position_exits:
                highway_exit = self.highway[self.position_exits[i]](current_outputs)
            # logits, pooled_output

            # inference stage
            if i in self.position_exits:
                if not self.training:
                    highway_logits = highway_exit[0]
                    # * entropy strategy
                    if self.exit_strategy == 'entropy':
                        highway_entropy = entropy(highway_logits)
                        highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                        all_highway_exits = all_highway_exits + (highway_exit,)

                        if highway_entropy < self.early_exit_threshold[self.position_exits[i]]:
                            # weight_func = lambda x: torch.exp(-3 * x) - 0.5**3
                            # weight_func = lambda x: 2 - torch.exp(x)
                            # weighted_logits = \
                            #     sum([weight_func(x[2]) * x[0] for x in all_highway_exits]) /\
                            #     sum([weight_func(x[2]) for x in all_highway_exits])
                            # new_output = (weighted_logits,) + current_outputs[1:] + (all_highway_exits,)
                            new_output = (highway_logits,) + current_outputs[1:] + ({"highway": all_highway_exits},)
                            raise HighwayException(new_output, i + 1)
                    # * confidence strategy
                    elif self.exit_strategy == 'confidence':
                        highway_confidence = confidence(highway_logits)
                        highway_exit = highway_exit + (highway_confidence,)
                        all_highway_exits = all_highway_exits + (highway_exit,)

                        if highway_confidence > self.early_exit_threshold[self.position_exits[i]]:
                            new_output = (highway_logits,) + current_outputs[1:] + ({"highway": all_highway_exits},)
                            raise HighwayException(new_output, i + 1)
                    # * patience strategy
                    elif self.exit_strategy == 'patience':
                        highway_prediction = prediction(highway_logits)
                        highway_exit = highway_exit + (highway_prediction,)
                        all_highway_exits = all_highway_exits + (highway_exit,)

                        if cnt == 0:
                            pred = highway_prediction
                            cnt += 1
                        else:
                            if pred == highway_prediction:
                                cnt +=1
                            else:
                                cnt = 1
                                pred = highway_prediction

                        if cnt == self.early_exit_threshold[0]:
                            new_output = (highway_logits,) + current_outputs[1:] + ({"highway": all_highway_exits},)
                            raise HighwayException(new_output, i + 1)
                    # * patient and confident strategy
                    elif self.exit_strategy == 'patient_and_confident':
                        highway_entropy = entropy(highway_logits)
                        highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                        all_highway_exits = all_highway_exits + (highway_exit,)

                        if highway_entropy < self.early_exit_threshold[self.position_exits[i]]:
                            pct += 1
                        else:
                            pct = 0

                        if pct == self.early_exit_threshold[-1]:
                            new_output = (highway_logits,) + current_outputs[1:] + ({"highway": all_highway_exits},)
                            raise HighwayException(new_output, i + 1)

                else:
                    all_highway_exits = all_highway_exits + (highway_exit,)


        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
            
        outputs = outputs + ({"highway": all_highway_exits},)
            
        return outputs


class DeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        # Pooler weights also needs to be loaded, especially in Highway!

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DeiTModel(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super(DeiTModel, self).__init__(config)
        self.config = config

        self.embeddings = DeiTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DeiTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = DeiTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def init_highway_pooler(self):
        self.encoder.init_highway_pooler(self.pooler)

    def get_input_embeddings(self) -> DeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)

        # retrun sequence_output, pooled_output, (hidden_states), (attentions), highway exits
        return head_outputs + encoder_outputs[1:]


class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start form 1!

class DeiTHighwayForImageClassification(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig, train_highway=True):
        super(DeiTHighwayForImageClassification, self).__init__(config)

        self.config = config
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.train_highway = train_highway
        self.exit_strategy = config.exit_strategy
        self.train_strategy = config.train_strategy
        self.loss_coefficient = config.loss_coefficient
        self.homo_loss_coefficient = config.homo_loss_coefficient
        self.hete_loss_coefficient = config.hete_loss_coefficient

        self.stage = 0 # for alternating training

        self.deit = DeiTModel(config)

        if config.backbone == 'ViT':
            self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        elif config.backbone == 'DeiT':
            self.cls_classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
            self.distillation_classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        else:
            raise ValueError("Please select one of the backbones: ViT, DeiT")
        
        self.position_exits = list(self.deit.encoder.position_exits)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            output_layer=-1,
    ):
        r'''
        lables:
        Outputs:
        Examples:
        '''
        exit_layer = self.num_layers
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        try:
            outputs = self.deit(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, )
            #  last_hidden_state(sequence_output), pooler_output, (hidden_states), (attentions), highway_exits

            sequence_output = outputs[0]

            if self.config.backbone == 'ViT':
                logits = self.classifier(sequence_output[:, 0, :])
            elif self.config.backbone == 'DeiT':
                cls_logits = self.cls_classifier(sequence_output[:, 0, :])
                distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])
                logits = (cls_logits + distillation_logits)/2
            
            outputs = (logits,) + outputs[2:]
            # logits, (hidden_states), (attentions), highway_exits(logits, pooler_output, entropy/confidence)

        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        if not self.training:
            if self.exit_strategy in ['entropy', 'patient_and_confident']:
                original_score = entropy(logits)
            elif self.exit_strategy == 'confidence':
                original_score = confidence(logits)
            elif self.exit_strategy == 'patience':
                original_score = prediction(logits)
            elif self.exit_strategy == 'confidence_and_entropy':
                original_score = (confidence(logits), entropy(logits))
            else:
                raise ValueError(
                    "Please select one of the exit strategies:entropy, confidence, patience, patient_and_confident")

            highway_score = []
            highway_logits_all = []

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            # work with highway exits
            highway_losses = []
            distillation_losses = []
            feature_losses = []

            if self.train_strategy in ['distillation', 'two-stage']:
                teacher_logits = logits

            for index, highway_exit in enumerate(outputs[-1]["highway"]):
                highway_logits = highway_exit[0]
                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_score.append(highway_exit[2])

                if self.config.problem_type == "regression":
                    # We are doing regression
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.squeeze(), labels.squeeze())
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    highway_loss = loss_fct(highway_logits, labels)

                if self.training and self.train_strategy == 'distillation':
                    
                    # * soft distillation
                    T = 2
                    highway_distill_loss = F.kl_div(
                        F.log_softmax(highway_logits / T, dim=1),
                        F.log_softmax(teacher_logits / T, dim=1),
                        reduction='sum',
                        log_target=True
                    ) * (T * T) / highway_logits.numel()
                    
                    # * hard distillation
                    # highway_distill_loss = F.cross_entropy(highway_logits.view(-1, self.num_labels),
                    #                                        teacher_logits.view(-1, self.num_labels).argmax(dim=1))

                    distillation_losses.append(highway_distill_loss)
                    

                highway_losses.append(highway_loss)

            if self.train_highway and self.training:
                if self.train_strategy in ['normal', ]:
                    outputs = ((sum(highway_losses)+loss) / (len(highway_losses)+1) ,) + outputs
                elif self.train_strategy in ['weighted', ]:
                    if self.config.num_early_exits == self.config.num_hidden_layers:
                        highway_losses = [highway_losses[index] * (coeff+1) for index, coeff in enumerate(self.position_exits)]
                        outputs = (sum(highway_losses) / sum(np.array(self.position_exits)+1),) + outputs
                    else:
                        highway_losses = [highway_losses[index] * (coeff + 1) for index, coeff in
                                          enumerate(self.position_exits)]
                        outputs = ((sum(highway_losses)+loss*12) / (sum(np.array(self.position_exits)+1) + 12),) + outputs
                elif self.train_strategy == 'alternating':
                    if self.stage % 2 == 0:
                        outputs = (loss,) + outputs
                    else:
                        outputs = ((sum(highway_losses)+loss) / (len(highway_losses)+1),) + outputs
                    self.stage += 1
                elif self.train_strategy == 'alternating_weighted':
                    if self.stage % 2 == 0:
                        outputs = (loss,) + outputs
                    else:
                        highway_losses = [highway_losses[index] * (coeff+1) for index, coeff in
                                          enumerate(self.position_exits)]
                        outputs = ((sum(highway_losses) + loss * 12) / (
                                sum(np.array(self.position_exits) + 1) + 12),) + outputs
                    self.stage +=1
                    # print(self.stage)
                elif self.train_strategy == 'distillation':
                    distill_coef = self.loss_coefficient
                    loss_all = (1 - distill_coef) * (sum(highway_losses) + loss) / (
                                len(highway_losses) + 1) + distill_coef * sum(
                        distillation_losses) / len(distillation_losses)
                    if output_hidden_states:
                        loss_all += sum(feature_losses) / len(feature_losses)
                    outputs = (loss_all,) + outputs
                else:
                    raise ValueError("Please select one of the training strategies:normal, weighted, alternating")

            else:
                outputs = (loss,) + outputs
                # loss, logits, highway_exits

        if not self.training:
            outputs = outputs[:-1] + ((original_score, highway_score), exit_layer)
            if output_layer >= 0:
                position = self.deit.encoder.position_exits[output_layer - 1]
                outputs = (outputs[0],) + (highway_logits_all[position],) + outputs[2:-1] + (output_layer,) ## use the highway of the last layer

        return outputs


class DeiTHighwayForImageClassification_distillation(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig, train_highway=False):
        super(DeiTHighwayForImageClassification_distillation, self).__init__(config)

        self.config = config
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.train_highway = train_highway
        self.exit_strategy = config.exit_strategy
        self.train_strategy = config.train_strategy
        self.loss_coefficient = config.loss_coefficient
        self.homo_coefficient = config.homo_coefficient
        self.hete_coefficient = config.hete_coefficient
        
        self.deit = DeiTModel(config)

        print(config.backbone)
        # Classifier head
        if config.backbone == 'ViT':
            self.classifier = nn.Linear(config.hidden_size,
                                        config.num_labels) if config.num_labels > 0 else nn.Identity()
        elif config.backbone == 'DeiT':
            self.cls_classifier = nn.Linear(config.hidden_size,
                                            config.num_labels) if config.num_labels > 0 else nn.Identity()
            self.distillation_classifier = nn.Linear(config.hidden_size,
                                                     config.num_labels) if config.num_labels > 0 else nn.Identity()
        else:
            raise ValueError("Please select one of the backbones: ViT, DeiT")

        # additional Components for aligning
        
        self.downsampling_r2 = DownSampling(2)
        self.downsampling_r3 = DownSampling(3)

        self.position_exits = list(self.deit.encoder.position_exits)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            output_layer=-1,
    ):
        r'''
        lables:
        Outputs:
        Examples:
        '''

        exit_layer = self.num_layers
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.deit.embeddings(pixel_values)

        if self.training:
            # freeze backbone
            hidden_states, hidden_list = embedding_output, []
            with torch.no_grad():
                for i in range(self.num_layers):
                    layer_head_mask = head_mask[i] if head_mask is not None else None
                    layer_outputs = self.deit.encoder.layer[i](hidden_states, layer_head_mask)
                    hidden_states = layer_outputs[0]
                    hidden_list.append((hidden_states,))
                    if 'distillation' in self.config.train_strategy:
                        sequence_output = hidden_list[-1][0]
                        if self.config.backbone == 'ViT':
                            teacher_logits = self.classifier(sequence_output[:, 0, :])
                        elif self.config.backbone == 'DeiT':
                            cls_logits = self.cls_classifier(sequence_output[:, 0, :])
                            distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])
                            teacher_logits = (cls_logits + distillation_logits) / 2

            if self.config.train_strategy == 'distillation':
                highway_losses = []
                distillation_losses = []
                all_highway_exits = ()
                for i in range(self.num_layers):
                    if i in self.position_exits:
                        index = self.deit.encoder.position_exits[i]
                        if self.config.encoder_ensemble == False:
                            highway_exit = self.deit.encoder.highway[index](hidden_list[i])
                        else:
                            cls_tokens = self.deit.encoder.highway[index](hidden_list[i])
                            if all_highway_exits == ():
                                highway_exit = self.deit.encoder.classifier[index](cls_tokens) + (cls_tokens,)
                            else:
                                cls_tokens = torch.concat((all_highway_exits[-1][2], cls_tokens), dim=1)
                                highway_exit = self.deit.encoder.classifier[index](cls_tokens) + (cls_tokens,)
                        all_highway_exits = all_highway_exits + (highway_exit,)
                        highway_logits = highway_exit[0]

                        loss_fct = CrossEntropyLoss()
                        highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                        highway_losses.append(highway_loss)

                        # * soft distillation
                        T = 2
                        highway_distill_loss = F.kl_div(
                            F.log_softmax(highway_logits / T, dim=1),
                            F.log_softmax(teacher_logits / T, dim=1),
                            reduction='sum',
                            log_target=True
                        ) * (T * T) / highway_logits.numel()

                        # * hard distillation
                        # highway_distill_loss = F.cross_entropy(highway_logits.view(-1, self.num_labels),
                        #                                        teacher_logits.view(-1, self.num_labels).argmax(dim=1))

                        distillation_losses.append(highway_distill_loss)
                distill_coef = self.loss_coefficient
                highway_losses = [highway_losses[index] * coeff for index, coeff in enumerate(self.position_exits)]

                loss_all = (1 - distill_coef) * sum(highway_losses) / sum(self.position_exits) + distill_coef * sum(
                    distillation_losses) / len(distillation_losses)
                outputs = (loss_all,)
                
            elif self.config.train_strategy == 'distillation_LGViT':
                highway_losses = []
                highway_distill_losses = []
                distillation_hete_losses = []
                distillation_losses_conv = []
                distillation_losses_attn = []
                all_highway_exits = ()
                
                distill_coef = self.loss_coefficient
                homo_distill_coef = self.homo_coefficient
                hete_distill_coef = self.hete_coefficient

                loss_fct = CrossEntropyLoss()

                # * prediction distillation loss
                for i in range(self.num_layers):
                    if i in self.position_exits:
                        index = self.deit.encoder.position_exits[i]
                        highway_exit = self.deit.encoder.highway[index](hidden_list[i])
                        all_highway_exits = all_highway_exits + (highway_exit,)
                        highway_logits = highway_exit[0]
                        highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                        highway_losses.append(highway_loss)

                        # * soft-distillation
                        T = 2
                        highway_distill_loss = F.kl_div(
                            F.log_softmax(highway_logits / T, dim=1),
                            F.log_softmax(teacher_logits / T, dim=1),
                            reduction='sum',
                            log_target=True
                        ) * (T * T) / highway_logits.numel()
                        highway_distill_losses.append(highway_distill_loss)

                
                distillation_pred_loss = (1-distill_coef) * sum(highway_losses) + distill_coef * sum(highway_distill_losses)

                # * heterogeneous distillation loss
                conv_feature_first = all_highway_exits[0][1]
                conv_feature_last = all_highway_exits[3][1]
                attn_feature_first = all_highway_exits[4][1]
                attn_feature_last = all_highway_exits[-1][1]
                hidden_states_last = hidden_list[-1][0][:, 2:, :]
                hidden_states_last_align_r3 = self.downsampling_r3(hidden_states_last)
                hidden_states_last_align_r2 = self.downsampling_r2(hidden_states_last)

                distillation_hete_losses.append(cal_hete_loss(conv_feature_first, hidden_states_last))
                distillation_hete_losses.append(cal_hete_loss(conv_feature_last, hidden_states_last))
                distillation_hete_losses.append(cal_hete_loss(attn_feature_last, hidden_states_last_align_r3))
                distillation_hete_losses.append(cal_hete_loss(attn_feature_first, hidden_states_last_align_r2))


                # * homogeneous distillation loss
                conv_feature_teacher = all_highway_exits[3][1]
                attn_feature_teacher = all_highway_exits[-1][1]

                for j in range(3):
                    distillation_losses_conv.append(
                        cal_homo_loss(all_highway_exits[j][1], conv_feature_teacher, loss_type='mse', align_type='none'))
                    if j == 2:
                        distillation_losses_attn.append(
                            cal_homo_loss(all_highway_exits[j + 4][1], attn_feature_teacher, loss_type='mse', align_type='inner'))
                    else:
                        distillation_losses_attn.append(
                            cal_homo_loss(all_highway_exits[j + 4][1], attn_feature_teacher,
                                          loss_type='mse', align_type='inner'))

                # * loss_all
                loss_all =  distillation_pred_loss + \
                            hete_distill_coef*sum(distillation_hete_losses) + \
                            homo_distill_coef*(sum(distillation_losses_conv)+ sum(distillation_losses_attn))

                outputs = (loss_all,)

        else:
            try:
                outputs = self.deit(
                    pixel_values,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict, )
                #  last_hidden_state(sequence_output), pooler_output, (hidden_states), (attentions), highway_exits

                sequence_output = outputs[0]

                if self.config.backbone == 'ViT':
                    logits = self.classifier(sequence_output[:, 0, :])
                elif self.config.backbone == 'DeiT':
                    cls_logits = self.cls_classifier(sequence_output[:, 0, :])
                    distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])
                    logits = (cls_logits + distillation_logits) / 2


                outputs = (logits,) + outputs[2:]
                # logits, (hidden_states), (attentions), highway_exits(logits, pooler_output, entropy/confidence)

            except HighwayException as e:
                outputs = e.message
                exit_layer = e.exit_layer
                logits = outputs[0]

            if self.exit_strategy in ['entropy', 'patient_and_confident']:
                original_score = entropy(logits)
            elif self.exit_strategy == 'confidence':
                original_score = confidence(logits)
            elif self.exit_strategy == 'patience':
                original_score = prediction(logits)
            else:
                raise ValueError(
                    "Please select one of the exit strategies:entropy, confidence, patience, patient_and_confident")

            loss = None
            highway_score = []
            highway_logits_all = []

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
            outputs = outputs[:-1] + ((original_score, highway_score), exit_layer)

        return outputs

class DownSampling(nn.Module):
    def __init__(self, sr_ratio):
        super().__init__()
        self.sampler = nn.AvgPool2d(1, sr_ratio)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.sampler(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def cal_homo_loss(stu_fea, tea_fea, loss_type='mse', align_type='none', mlp=None):
    '''
        loss for homogeneous distillation
    '''

    if align_type == 'none':
        pass
    if align_type == 'inner':
        # stu_fea = F.normalize(stu_fea, dim=-1)
        stu_fea = stu_fea.transpose(-1, -2) @ stu_fea
        # tea_fea = F.normalize(tea_fea, dim=-1)
        tea_fea = tea_fea.transpose(-1, -2) @ tea_fea

    if loss_type == 'kldiv':
        stu_fea = F.log_softmax(stu_fea, dim=-1)
        tea_fea = F.log_softmax(tea_fea, dim=-1)
        loss = F.kl_div(stu_fea, tea_fea, reduction="batchmean", log_target=True)
    elif loss_type == 'mse':
        loss = F.mse_loss(stu_fea, tea_fea)
    else:
        raise ValueError('Please select one of the loss_type: kldiv, mse')
    
    return loss

def cal_hete_loss(stu_fea, tea_fea):
    '''
    loss for heterogeneous distillation
    '''

    stu_fea = F.log_softmax(stu_fea, dim=-1)
    tea_fea = F.log_softmax(tea_fea, dim=-1)
    loss = F.kl_div(stu_fea, tea_fea, reduction="batchmean", log_target=True)
    
    return loss
