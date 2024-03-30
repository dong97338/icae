# ICAE Code

## Updated Version (V2, March 2024):

1. Two Mistral-based checkpoints along with corresponding inference code have been uploaded in the 'icae_v2/' folder.

2. For our V2 models, please use the official PEFT package and install Transformers version 4.36.2.

## Original Version (V1, September 2023):

1. To reproduce our results, the use of our customized PEFT package is required. Install it by running the following command: "pip install -e peft/"

2. The core code for In-Context Autoencoder (ICAE) is located in the 'icae/' directory. Within this directory, 'utils/stable_trainer.py' is used for token-level (micro-average) loss during training. If you are not concerned with token-level or sample-level details, this can be ignored and you may utilize the default 'trainer.py'.  

3. Transformers version 4.31.0 is necessary. Please note that earlier versions of Transformers are not supported.