# Megafon course work

## [Link to Kaggle notebook](https://www.kaggle.com/konstantinalbul/course-work)

## Used stack

* ML: sklearn, pandas, numpy, dask

### Model: GradientBoostingClassifier

### Metric: [f1_macro](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score)

## Description

The goal is to predict the probability will buy user service or not.

## Data

* target - target feature, where 1 means user buy service, 0 - not
* buy_time - buy time in Timestamp format
* id - user id
* vas_id - service that user buy
* 0-252 - user features

## How to run

### The test dataset should be in the directory: `./data/data_test.csv`

### Install venv and activate it

```bash
python -m venv venv

# On Windows
source venv/Scripts/activate

# On Unix or MacOS
source venv/bin/activate
```

### Install requirments

```bash
pip install -r requirments.txt
```

### Run

```bash
python megafon.py
```

### After run script will be created file `answers_test.csv` in root directory with predictions
