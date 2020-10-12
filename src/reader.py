from typing import Iterable, Dict

import jsonlines
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer

@DatasetReader.register('symbol_sent')
class SymbolSentimentDatasetReader(DatasetReader):

    def __init__(self, tokenizer:Tokenizer = None, token_indexers:Dict[str, TokenIndexer] =None, lazy:bool=True):

        super().__init__(lazy)
        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {'tokens':SingleIdTokenIndexer()}


    def _read(self, file_path: str) -> Iterable[Instance]:
        with jsonlines.open(file_path) as fp:
            for line in fp:
                yield self.text_to_instance(line)

    def text_to_instance(self, jsonline) -> Instance:
        text = jsonline['text']
        label = jsonline['label']
        instance_dict  = {}
        instance_dict['tokens'] = TextField(self._tokenizer.tokenize(text), token_indexers=self._token_indexers)
        instance_dict['label'] = LabelField(label, skip_indexing=True)
        instance_dict['metadata'] = MetadataField({'id':jsonline['id'],'text':text})
        return Instance(instance_dict)