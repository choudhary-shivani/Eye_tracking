# Eye_tracking

This repo is submission from my team for the Eye tracking competition. The link to the competition organizers is:
https://cmclorg.github.io/

This repo provides the files to tag the data. Use the provided script. Analysis.py to generate the files with the BERT small embedding. Each of the words are treated as separate sentence for the process of extraction of embedding.

## CNN
Files CNN.py, traintf.py provides the details of the DL model I used to extract the emdeddings. It will train separate models for the each of the field. During the inference time. It will load each modules separately and prepare the inference output. 

For CNN, there is flag that will ensure what would be the context length for the processing.

Added the modules that was used to train on the colab. 

Added modules to hypertune

BASE CNN model : CNN_colab.py
Hypertunning model: Optimize.py
Annopti.py

## LGBM
The file Data Preparation LGBM.py provides the steps used for preparing the data and extracting features for training the model. Different combinations of features can be extracted into numpy arrays for forming various models.

The file LGBM Regressor_syll_bert.py provides the details of the model applied to the feature space consisting of syllables and BERT embeddings.

The file LGBM Regressor_syll_wid_bert.py provides the details of the model applied to the feature space consisting of syllables, word_id and BERT embeddings.

The file LGBM Regressor_syll_wid_wlen_bert.py provides the details of the model applied to the feature space consisting of syllables, word_id, word_length and BERT embeddings.

The file LGBM Regressor_syll_wlen_bert.py provides the details of the model applied to the feature space consisting of syllables, word_length and BERT embeddings.

The file LGBM Regressor_wid_wlen.py provides the details of the model applied to the feature space consisting of word_id, word_length, and BERT embeddings.

LGBM regressors were trained on google colaboratory

