# Overview
The goal of this [competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) is to predict which clients are more likely to **default on their loans**. The evaluation will favor solutions that are **stable over time**.

# Setup
1. Create a folder `data` with the parquet files from the [Kaggle competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data?select=parquet_files) 
2. Create a virtualenv `venv` (or whatever name you want but remember to add it to the gitignore!)
3. Run `pip install -r /path/to/requirements.txt`

# Repo structure
- Notebooks are stored in the `notebooks` folder, currently exploring the dataset.
- Insights, questions, and granular todo's are currently stored in the `doc` folder.
- `src` contains useful functions to be used in all notebooks. This is done to improve code versioning.

# TODO's
- ~~Repo setup~~
- Explore datasets and gather insights in the `doc` folder
