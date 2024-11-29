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

    train, test = train_test_split(data, test_size=0.25, random_state=42)
    X_train, y_train = train.drop(["quality"], axis=1), train[["quality"]]
    X_test, y_test = test.drop(["quality"], axis=1), test[["quality"]]
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, y_pred)

        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)
