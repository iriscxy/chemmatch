# Code and data for paper "Unveiling the Power of Language Models in Chemical Research Question Answering".

## Project Overview

This project provides code for data collection and model training/testing. The folder structure includes two main directories:

- **`code`**: Contains scripts for data collection and model training/evaluation.
- **`data`**: Includes `test.json`, a JSON file with test data, and `model_best.pth`, a pre-trained model checkpoint.

## Data Collection Code

The data collection scripts, located in `code/data_collection_code`, are designed to collect data from five different websites. Each script is numbered to indicate the order in which it should be run. 

1. **Elsevier** (`elsevier/`): 
   - `1cursor.py`: Initial data fetching.
   - `2extract.py`: Data extraction.

2. **Lens** (`lens/`): 
   - `1lens_cursor_bio.py`, `1lens_cursor_cata.py`, `1lens_cursor_elec.py`, `1lens_cursor_enginner.py`: These scripts are for fetching data across different categories (e.g., biology, catalog, electronics, engineering).
   
3. **S2ORC** (`s2orc/`):
   - `1abstracts.py`: Fetch abstracts.
   - `1s2orc.py`: Main data fetching.
   - `2extract_paper.py`: Paper extraction.

4. **Scopus** (`scopus/`): 
   - `1cursor.py`: Initial data fetching.
   - `2extract.py`: Data extraction.

5. **Springer** (`springer/`):
   - `1cursor.py`: Initial data fetching.
   - `2extract.py`: Data extraction.
   
**Note**: To collect data, you need API keys for the respective websites.


## Model Code

The model code, located in `code/model_code`, includes all necessary files for training and evaluating the model. Follow the steps below to set up the environment, preprocess the data, train the model, and evaluate it.

### 1. Setup Environment

Install dependencies listed in `requirements.txt`:

```bash
pip install -r code/model_code/requirements.txt
```

### 2. Data Preprocessing
After collecting the dataset, use the code in the `preprocess/set1_ver1` folder to format the data as required by the model. This step will clean, organize, and transform the raw data into the structure needed for training and evaluation.

   - **For test data**: Run `generate_json_test.py` to process test data.
   - **For labeled training data**: Run `generate_json_train_label.py` to process labeled training data.
   - **For unlabeled training data**: Run `generate_json_train_unlabel.py` to process unlabeled training data.

### 3. Training

Use `train.sh` to train the model. This script will initiate the training process with the preprocessed data:

```bash
bash code/model_code/train.sh
```

### 4. Evaluation

After training, use `evaluate.sh` to evaluate the model on the test data:

```bash
bash code/model_code/evaluate.sh
```

We also provide one checkpoint file in [this link](https://drive.google.com/file/d/15TbE3_yGzCIV5OKwoxsBFvroinNwm8nk/view?usp=sharing).