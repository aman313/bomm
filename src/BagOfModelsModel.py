from typing import Dict, Optional

import torch
from allennlp.common import Params, Registrable
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import ModuleList, Linear
from torch.nn.functional import nll_loss


class TauScheduler():

    def __init__(self, tau_start = 10, tau_end = 0.1, step=100):
        pass


TAU=10.0
is_hard_softmax = False

class BagOfModelSeq2VecEncoder(Seq2VecEncoder):

    def __init__(self, base_encoder_name:str, base_encoder_params:Params,bag_size:int):
        super().__init__()
        self._model_bag = ModuleList()
        self._bag_size = bag_size
        self._model_class = Seq2VecEncoder.by_name(base_encoder_name)
        self._base_params = base_encoder_params

        for i in range(bag_size):
            m = self._model_class.from_params(Params(base_encoder_params))
            initializer = InitializerApplicator()
            initializer(m)
            self._model_bag.append(m)
        self._output_dim = self._model_bag[0].get_output_dim()
        self._input_dim = self._model_bag[0].get_input_dim()
        self._param_keys = [x[0]  for x in self._model_bag[0].named_parameters()]

    def get_weighted_weights(self, model_weights):
        state_dict_new = {}
        model_weights = torch.nn.Softmax()(model_weights)
        for key in self._param_keys:
            for i in range(self._bag_size):
                try:
                    state_dict_new[key] += model_weights[0][i] * {x[0]:x[1] for x in self._model_bag[i].named_parameters()}[key]
                except KeyError:
                    state_dict_new[key] = model_weights[0][i] * {x[0]:x[1] for x in self._model_bag[i].named_parameters()}[key]
        #weighted_model = self._model_class.from_params(Params(self._base_params))
        #weighted_model.load_state_dict(state_dict_new)
        #for param in weighted_model.parameters():
        #    param.requires_grad = False
        return state_dict_new


    def functional_model(self, weighted_weights, input, mask):
        raise NotImplementedError

    def get_weighted_output(self, model_outputs, model_weights):
        #predictions = 0
        model_weights = torch.nn.functional.gumbel_softmax(model_weights, hard=is_hard_softmax, tau=TAU) # B x BS
        #print(model_weights)
        model_outputs = torch.cat([x.unsqueeze(0) for x in model_outputs]) # BS X B  X D
        model_outputs = model_outputs.permute((1,2,0)) # B  X D X BS
        predictions = model_outputs.bmm(model_weights.unsqueeze(-1)) #B X D X 1
        #for i in range(self._bag_size):
        #    predictions += model_weights[0][i]*model_outputs[i]
        return predictions.squeeze(-1)

    def forward(self, input, model_weights,mask):
        # input size: batch_size X seq_len X dim
        # model_weights: batch_size X num_models

        #weighted_weights = self.get_weighted_weights(model_weights)
        prediction_bag = [self._model_bag[i](input,mask) for i in range(self._bag_size)]
        predictions =  self.get_weighted_output(prediction_bag,model_weights)
        return  predictions


    def get_output_dim(self) -> int:
        return self._output_dim

    def get_input_dim(self) -> int:
        return self._input_dim

@Model.register('bom_tcm')
class BagOfModelsTextClassifier(Model):

    def __validate_bag__(self):
        raise NotImplementedError

    def __init__(self, vocab:Vocabulary, class_predictor:BagOfModelSeq2VecEncoder, text_field_embedder:TextFieldEmbedder, model_chooser:Seq2VecEncoder,num_class:int,
                 initializer:InitializerApplicator=InitializerApplicator(),regularizer:Optional[RegularizerApplicator] = None ):
        super().__init__(vocab,regularizer)
        self._text_field_embedder = text_field_embedder
        self._class_predictor = class_predictor
        self._model_chooser = model_chooser # Seq2VecEncoder model choosing does not work for batch sizes > 1
        self._num_class = num_class
        self._projector = Linear(self._class_predictor.get_output_dim(),num_class)
        self._model_projector = Linear(self._model_chooser.get_output_dim(),self._class_predictor._bag_size)
        self._accuracy = CategoricalAccuracy()
        initializer(self)


    def forward(self, tokens, label=None, metadata=None) -> Dict[str, torch.Tensor]:
        embedded_tokens = self._text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens).float()

        model_weights = self._model_chooser(embedded_tokens,mask)
        model_weights = self._model_projector(model_weights)
        '''
        weight_mat = []
        for i in range(len(metadata)):
            if len(set(['0','1','2','3','4','5']).intersection(set(metadata[i]['text']))) > 0:
                weight_mat.append([1.0,0.0,0.0])
            elif len(set(['a','b','c','d','e','f']).intersection(set(metadata[i]['text']))) > 0:
                weight_mat.append([0.0,1.0,0.0])
            else:
                weight_mat.append([0.0,0.0,1.0])
        model_weights = torch.tensor(weight_mat,device='cuda:0')
        '''
        predictions = self._projector(self._class_predictor(embedded_tokens,model_weights,mask ))
        predictions = torch.nn.LogSoftmax()(predictions)
        output_dict = {}

        if label is not None:
            loss = nll_loss(predictions,label)
            output_dict['loss'] = loss
            output_dict['labels'] = label
            self._accuracy(predictions,label)
        if metadata is not None:
            output_dict['metadata'] = metadata
        output_dict['predictions'] = predictions
        output_dict['pred_probs'] = torch.nn.Softmax()(predictions)
        output_dict['weights'] = torch.nn.functional.gumbel_softmax(model_weights, hard=is_hard_softmax, tau=TAU)
        return output_dict



    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics