training:
allennlp train symbolsent_classification_config.jsonnet -s <run_dir> --include-package src

prediction:
allennlp predict --output-file <pred_out> <path-to-model.tar.gz> <test-file-location> --include-package src --predictor jsonline_dataset_predictor

Data and wordvec generation:

refer main block of utils.py