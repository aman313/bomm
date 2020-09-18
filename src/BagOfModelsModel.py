from typing import Dict, Optional

import torch
from allennlp.common import Params, Registrable
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from torch.nn import ModuleList
from torch.nn.functional import nll_loss


class BagOfModelSeq2VecEncoder(Seq2VecEncoder):

    def __init__(self, base_encoder_name:str, base_encoder_params:Params,bag_size:int):
        super().__init__()
        self._model_bag = ModuleList()
        self._bag_size = bag_size
        model_class = Seq2VecEncoder.by_name(base_encoder_name)
        for i in range(bag_size):
            self._model_bag.append(model_class.from_params(Params(base_encoder_params)))
        self._param_keys = [x[0]  for x in self._model_bag[0].named_parameters()]

    def get_weighted_model(self, model_weights):
        state_dict_new = {}
        model_weights = torch.nn.Softmax()(model_weights)
        for key in self._param_keys:
            for i in range(self._bag_size):
                try:
                    state_dict_new[key] += model_weights[i] * self.__model_bag[i].named_paraeters()[key]
                except KeyError:
                    state_dict_new[key] = model_weights[i] * self.__model_bag[i].named_parameters()[key]
        weighted_model = torch.nn.Module.load_state_dict(state_dict_new)
        return weighted_model

    def forward(self, input, model_weights):
        # input size: batch_size X seq_len X dim
        # model_weights: batch_size X num_models

        weighted_model = self.get_weighted_model(model_weights)
        predictions =  weighted_model(input)
        return  predictions


@Model.register('bom_tcm')
class BagOfModelsTextClassifier(Model):

    def __validate_bag__(self):
        raise NotImplementedError

    def __init__(self, vocab:Vocabulary, class_predictor:BagOfModelSeq2VecEncoder, text_field_embedder:TextFieldEmbedder, model_chooser:Seq2VecEncoder,
                 initializer:InitializerApplicator=InitializerApplicator(),regularizer:Optional[RegularizerApplicator] = None ):
        super().__init__(vocab,regularizer)
        self._text_field_embedder = text_field_embedder
        self._class_predictor = class_predictor
        self._model_chooser = model_chooser # Seq2VecEncoder model choosing does not work for batch sizes > 1
        initializer(self)

    def get_mixed_models(self):
        raise NotImplementedError


    def get_predictions(self):
        raise NotImplementedError

    def forward(self, tokens, label) -> Dict[str, torch.Tensor]:
        embedded_tokens = self._text_field_embedder(tokens)
        model_weights = self._model_chooser(embedded_tokens,util.get_text_field_mask(tokens).float())
        predictions = self._class_predictor(embedded_tokens,model_weights)

        output_dict = {}
        if label:
            loss = nll_loss(predictions,label)
            output_dict['loss'] = loss
        output_dict['predictions'] = predictions
        return output_dict

