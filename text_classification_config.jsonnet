 {
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "5-class"
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "5-class"
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
  "model": {
    "type": "bom_tcm",
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
    "class_predictor": {
        "base_encoder_name":"lstm",
        "base_encoder_params":{
            "input_size":100,
            "hidden_size":50,
            "num_layers":2,
            "bidirectional":true,
            "dropout":0.1

        },
        "bag_size":3

    },
    "model_chooser": {
        "type":"lstm",
        "input_size":100,
         "hidden_size":3,
            "num_layers":2,
            "bidirectional":false,
            "dropout":0.1
    }

  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 1
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "-loss",
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}