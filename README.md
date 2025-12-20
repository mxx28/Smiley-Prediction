# <img src="twitter_icon.png" alt="Twitter icon" width="25" /> Smiley Prediction in Twitter Sentiment Data
The task of this competition is to predict whether a tweet originally contained a positive `:)` or a negative `:(` smiley, using only the remaining text.  

Our final results were submitted to [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-text-classification), where our best model achieved **92.0%** accuracy. 

### Overall test accuracy on AIcrowd

| Model                | Description                          | Test Accuracy (%) | Submission ID |
|----------------------|--------------------------------------|-------------------|---------------|
| GloVe                | GloVe + MLP (improved)               | 82.9              | #304285       |
| FastText             | FastText + MLP (improved)            | 83.6              | #304286       |
| TweetNLP-sentiment   | TweetNLP-sentiment (fine-tuned)      | 90.6              | #304369       |
| BERTweet             | BERTweet (fine-tuned)                | 92.0              | #305590       |


Further details and illustrations are provided in the accompanying report.

## Data (`twitter-datasets/`)

Download the dataset from the official [AIcrowd data page](https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files)

We assume the dataset is placed in a folder named `twitter-datasets/` at the project root.

## GloVe-based model (`Glove/`)

This folder contains the implementation of our GloVe-based baseline and its improved variants for the tweet sentiment task. It covers the whole pipeline from building the vocabulary and co-occurrence matrix, training [GloVe embeddings](https://aclanthology.org/D14-1162/), constructing tweet-level features, to training the classifier and generating a submission file.

### Requirements

```bash
pip install numpy scipy scikit-learn joblib
```


### Main scripts

- `build_vocab.sh`  
  Preprocesses the raw tweets and builds the initial vocabulary files (`vocab_full.txt`, `vocab.pkl`).

- `cut_vocab.sh`  
  Prunes the vocabulary according to minimum frequency / size constraints and writes the reduced vocabulary to `vocab_cut.txt`.

- `cooc.py`  
  Constructs the word–word co-occurrence matrix from the tokenised tweets (using the local context window) and saves it as `cooc.pkl`.

- `glove_solution_mini_batch.py`  
  Trains GloVe embeddings from `cooc.pkl` via mini-batch optimisation and stores the resulting matrix in `embeddings.npy`.

- `build_features.py`  
  Converts tweets into feature vectors using the learned embeddings (e.g. mean+max pooling), and saves them as `train_features.npy` together with labels `train_labels.npy`.

- `train_classifier.py`  
  Trains the logistic/MLP classifier on the feature matrix and stores the fitted model in `mlp_model.joblib`.

- `predict_test.py`  
  Applies the trained classifier to the test set features and writes the predictions to `submission_glove.csv`.

- `run_GloVe.ipynb`  
  Jupyter notebook that runs the end-to-end GloVe pipeline and was used for exploratory experiments.

### Generated / data files

- `cooc.pkl` – serialized co-occurrence matrix.  
- `embeddings.npy` – learned GloVe word embeddings.  
- `train_features.npy`, `train_labels.npy` – cached training features and labels.  
- `vocab_full.txt`, `vocab_cut.txt`, `vocab.pkl` – vocabulary files.  
- `submission_glove.csv` – submission file produced by the GloVe-based model.

### Typical usage

1. From the project root:

```bash
cd GloVe
bash build_vocab.sh
bash cut_vocab.sh          
python cooc.py
python glove_solution_mini_batch.py
python build_features.py
python train_classifier.py
python predict_test.py    
```

2. Follow the notebook `run_GloVe.ipynb`  

## FastText-based model (`FastText/`)

This folder contains the implementation of our FastText-based model for the tweet sentiment task. It covers the whole pipeline from merging training data, training unsupervised [FastText embeddings](https://fasttext.cc/), constructing tweet-level features using mean+max pooling, to training the MLP classifier and generating a submission file.

### Requirements

```bash
pip install numpy fasttext scikit-learn joblib
```

### Main scripts

- `build_all_train_txt.py`  
  Merges positive and negative training tweets into a single corpus file (`all_train.txt`) for unsupervised FastText training.

- `train_fasttext_unsup.py`  
  Trains unsupervised FastText word embeddings from the merged corpus using skipgram or CBOW and saves the model as `twitter_fasttext.bin`.

- `build_features_fasttext.py`  
  Converts tweets into feature vectors using the learned FastText embeddings (mean+max pooling), and saves them as `train_features_fasttext.npy` together with labels `train_labels_fasttext.npy`.

- `train_mlp_fasttext.py`  
  Trains the MLP classifier on the FastText feature matrix with hyperparameter tuning and stores the fitted model in `mlp_fasttext_model.joblib`.

- `predict_fasttext.py`  
  Applies the trained classifier to the test set features and writes the predictions to `submission_fasttext.csv`.

- `run_FastText.ipynb`  
  Jupyter notebook that runs the end-to-end FastText pipeline and was used for exploratory experiments.

### Generated / data files

- `twitter_fasttext.bin` – trained FastText word embeddings model.  
- `train_features_fasttext.npy`, `train_labels_fasttext.npy` – cached training features and labels.  
- `mlp_fasttext_model.joblib` – trained MLP classifier model.  
- `submission_fasttext.csv` – submission file produced by the FastText-based model.

### Typical usage

1. From the project root:

```bash
cd FastText
python build_all_train_txt.py
python train_fasttext_unsup.py
python build_features_fasttext.py
python train_mlp_fasttext.py
python predict_fasttext.py 
```

2. Follow the notebook `run_FastText.ipynb`  

## Pretrained large language model (`TweetNLP-sentiment/`)
This folder contains the implemtation of fine-tuning the pre-trained large language model [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), and see [TweetNLP](https://tweetnlp.org/resources/) for more details. After fine-tuned with the provided data, the model will generate a submission file based on the test data.

We fine-tune the entire encoder (including embeddings and transformer layers), not only the classifier head.

Because the original model is trained for 3-class sentiment, we used 2 methods:
- Replace the classification head with a randomly initialized 2-class head.
- Train with the 3-class sentiment and output pos if pos prob > neg prob, otherwise output neg.

Detailed explanations are provided in the report. This folder reproduces the **0.906** accuracy result, and the trained model is publicly available on [Hugging Face](https://huggingface.co/MengRen46/TweetNLP-sentiment)

### Requirements
This project is implemented in Python using the HuggingFace Transformers and PyTorch ecosystems.
To ensure reproducibility, we provide a minimal requirements.txt that includes all libraries needed for training, evaluating, and running inference.

```bash
pip install -r requirements.txt
```

### Main scripts
- **`train.py`**  
  Fine-tunes a pre-trained Twitter-RoBERTa model on the cleaned training dataset.  
  The preprocessing stage removes noisy or non-informative tokens (e.g., `<user>`, URLs) while preserving sentiment-relevant text.  
  After training, the fine-tuned model and tokenizer are saved in the `model_out` directory.

- **`predict.py`**  
  Loads the fine-tuned model and performs inference on the test dataset.  
  It generates sentiment predictions for each tweet and writes the final results to `submission.csv` in the required output format.

- **`run.py`**  
  Downloads the fine-tuned model from Hugging Face (default configured to `MengRen46/TweetNLP-sentiment`), performs batched inference on the test set, and generates `submission.csv`.

### Typical usage

From the project root:

1. Train the model by yourself
```bash
cd TweetNLP-sentiment
python train.py
python predict.py
```

2. Use our fine-tuned model
```bash
cd TweetNLP-sentiment
python run.py
```




## BERTweet models (`BERTweet/`)

This folder contains the implementation of our best-performing models using [BERTweet](https://github.com/VinAIResearch/BERTweet), a RoBERTa-based model pre-trained on English Tweets. This approach achieved **92.0%** accuracy on the test set.

We provide scripts for fine-tuning `bertweet-large` using standard techniques and `bertweet-base` using Layer-wise Learning Rate Decay (LLRD).

### Requirements

```bash
pip install torch transformers datasets scikit-learn huggingface_hub
```

### Main scripts

- **`train.py`**  
Fine-tunes the `vinai/bertweet-large` model. It handles data loading, tokenization with caching, and training using the HuggingFace `Trainer` API. It saves the best model based on accuracy.

- **`train_lldr.py`**  
  Fine-tunes the `vinai/bertweet-base` model using Layer-wise Learning Rate Decay (LLRD). This script implements custom optimizer grouping to apply different learning rates to the classifier, encoder layers, and embeddings, along with specific text cleaning and label smoothing.

- **`upload_model_to_hf.py`**  
A utility script to upload the trained checkpoint to the Hugging Face Hub.

- **`run.py`**  
  Downloads the fine-tuned model from Hugging Face (default configured to `RangerX/Bertweet_large`), performs batched inference on the test set, and generates `submission_large.csv`.


### Typical usage

From the project root:

1. Train the model
```bash
cd BERTweet
python train.py
```
2. Upload to Hugging Face for remote access
```bash
cd BERTweet
python upload_model_to_hf.py
```

3. Generate predictions
```bash
cd BERTweet
python run.py
```
