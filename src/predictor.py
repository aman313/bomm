from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor


@Predictor.register('jsonline_dataset_predictor')
class JsonlineDatasetReaderPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance =  self._dataset_reader.text_to_instance(json_dict)
        if not instance:
            raise Exception('Could not convert to instances', json_dict)
        return  instance

    def predictions_to_labeled_instances(self, instance, outputs):
        raise NotImplementedError