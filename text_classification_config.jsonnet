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
    "type": "bom_tcm",
    "num_class":2,
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
        "bag_size":20

    },
    "model_chooser": {
        "type":"lstm",
        "input_size":100,
         "hidden_size":20,
            "num_layers":2,
            "bidirectional":false,
            "dropout":0.1
    }

  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 10
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 20,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}