import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor on the train dataset
    :param x_train: (DataFrame) containing the train features
    :param y_train: (Series) containing the train target
    :return: (RandomForestRegressor) trained model
    """
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=20, min_samples_split=10, n_jobs=-1)
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

