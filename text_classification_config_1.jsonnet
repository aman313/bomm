 {
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "2-class"
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class"
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
            "type": "embedding",
            "embedding_dim": 100,
            "trainable": false
        }
      }
    },
    "seq2vec_encoder": {
        "type":"lstm",
            "input_size":100,
            "hidden_size":10,
            "num_layers":5,
            "bidirectional":true,
            "dropout":0.1

        },
     "num_labels":2


  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 30
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 20,
    "grad_norm": 5.0,
    "validation_metric": "-loss",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}