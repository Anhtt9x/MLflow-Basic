import os
import warnings
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import logging
import mlflow 
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import dagshub
dagshub.init(repo_owner='Anhtt9x', repo_name='MLflow-Basic', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(true, pred):
    rmse = np.sqrt(mean_squared_error(true,pred))
    mae = mean_absolute_error(true,pred)
    r2 = r2_score(true,pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)  
    # Load data
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/tests/datasets/winequality-red.csv"

    try:
        data = pd.read_csv(csv_url,sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check url: %s", e)

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train = train.drop(["quality"], axis=1), train[["quality"]]
    X_test, y_test = test.drop(["quality"], axis=1), test[["quality"]]
    