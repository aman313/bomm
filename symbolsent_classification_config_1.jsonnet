 {
  "dataset_reader":{
    "type": "symbol_sent"
  },
  "validation_dataset_reader":{
    "type": "symbol_sent"

  },
  "train_data_path": "/media/aman/8a3ffbda-8a36-45f9-b426-d146a65d9ece/data1/research/bomm/symbol_sent.train",
  "validation_data_path": "/media/aman/8a3ffbda-8a36-45f9-b426-d146a65d9ece/data1/research/bomm/symbol_sent.dev",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 16,
            "pretrained_file":"/media/aman/8a3ffbda-8a36-45f9-b426-d146a65d9ece/data1/research/bomm/symbol_sent_wv.txt",
            "trainable":false

        }
      }
    },
    "seq2vec_encoder": {
        "type":"lstm",
            "input_size":16,
            "hidden_size":50,
            "num_layers":5,
            "bidirectional":true,
            "dropout":0.1

        },
     "num_labels":3


  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 30
  },
  "trainer": {
    "num_epochs": 200,
    "patience": 200,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}