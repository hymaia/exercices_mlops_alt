import pandas as pd
import joblib
from src.gathering.task import DataCollector
from src.gathering.cleaning import DataCleaner
from src.features.task import FeaturesEngineering
from src.ml.task import train_model
from src.ml.validation import split_train_and_val_sets, compute_metrics


def main():
    """
    Loads, prepares data and trains a model. All artefacts are saved in the models folder
    :return:
    """
    # load data and split train set
    path_train_set = "../data/raw/train.csv"
    path_features_set = "../data/raw/features.csv"
    path_stores_set = "../data/raw/stores.csv"
    df_train = DataCollector().gather_data(path_train_set, path_features_set, path_stores_set)
    x_train, x_val, y_train, y_val = split_train_and_val_sets(df_train)

    # clean data
    cleaner = DataCleaner().fit(df_train)
    x_train = cleaner.transform(x_train)
    x_val = cleaner.transform(x_val)

    # features engineering
    features_transformer = FeaturesEngineering().fit(x_train, y_train)
    x_train = features_transformer.transform(x_train)
    x_val = features_transformer.transform(x_val)
    x_train.to_excel("../data/processed/x_train_processed.xlsx", index=False)
    x_val.to_excel("../data/processed/x_val_processed.xlsx", index=False)

    # train model
    model = train_model(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)

    # display metrics
    print("Train set :")
    compute_metrics(y_train, pred_train)
    print("Validation set :")
    compute_metrics(y_val, pred_val)

    # save predictions and artefacts
    pd.DataFrame(pred_train).to_excel("../data/processed/pred_train.xlsx", index=False)
    pd.DataFrame(pred_val).to_excel("../data/processed/pred_val.xlsx", index=False)
    joblib.dump(model, "../models/model.pkl")

    # save cleaner et features_transformer
    joblib.dump(cleaner, "../models/cleaner.pkl")
    joblib.dump(features_transformer, "../models/features_transformer.pkl")


if __name__ == "__main__":
    main()
