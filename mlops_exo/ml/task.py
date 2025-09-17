import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(
    x_train: pd.DataFrame, y_train: pd.Series, dict_params: dict
) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor on the train dataset
    :param x_train: (DataFrame) containing the train features
    :param y_train: (Series) containing the train target
    :return: (RandomForestRegressor) trained model
    """
    # features selection
    print("features selection")

    print("training random forest")

    # print example
    print("observation example")
    print(x_train.iloc[0].to_json())

    # entrainement du modèle
    model = RandomForestRegressor(
        n_estimators=dict_params["n_estimators"],
        random_state=dict_params["random_state"],
        max_depth=dict_params["max_depth"],
        min_samples_split=dict_params["min_samples_split"],
        n_jobs=dict_params["n_jobs"],
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
    # x_test = x_test.drop(columns=["Date", "IsHoliday", "Type"])
    return model.predict(x_test)
