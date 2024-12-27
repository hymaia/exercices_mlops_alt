import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor on the train dataset
    :param x_train: (DataFrame) containing the train features
    :param y_train: (Series) containing the train target
    :return: (RandomForestRegressor) trained model
    """
    n_estimators = 10
    max_depth = 20
    min_samples_split = 10
    random_state = 42

    # TODO : exercice 3.3 : ajoutez les hyper-paramètres dans MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("random_state", random_state)

    # entrainement du modèle
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, max_depth=max_depth, min_samples_split=min_samples_split, n_jobs=-1
    )
    model.fit(x_train, y_train)
    return model


def predict_with_model(x_test: pd.DataFrame, model: RandomForestRegressor):
    """
    Generates predictions on the test dataset
    :param x_test: (DataFrame) containing the test features
    :param model: (RandomForestRegressor) trained model
    :return: (Series) containing the predictions
    """
    return model.predict(x_test)
