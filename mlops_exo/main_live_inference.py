import pandas as pd
import requests
import joblib
import os
import sys
import warnings
import mlflow
import time
from config import DATA_RAW
warnings.filterwarnings("ignore")


##########################################################
# PART 1 : inference with all features already computes  #
##########################################################

print("----- PARTIE 1 : prédiction avec toutes les features déjà données")
data_json_1 = {
    "dataframe_records": [
        {
            "Store": 8986.8302954228,
            "Dept": 11883.1919251024,
            "Temperature": 30.34,
            "Fuel_Price": 3.811,
            "MarkDown1": 0.0,
            "MarkDown2": 0.0,
            "MarkDown3": 0.0,
            "MarkDown4": 0.0,
            "MarkDown5": 0.0,
            "CPI": 134.0682581,
            "Unemployment": 7.658,
            "Size": 123737.0,
            "MarkdownsSum": 0.0,
            "Day": 1.0,
            "Week": 13.0,
            "Month": 4.0,
            "Year": 2011.0,
            "Days_to_Thansksgiving": 237.0,
            "Days_to_Christmas": 267.0,
            "SuperBowlWeek": 0.0,
            "LaborDayWeek": 0.0,
            "ThanksgivingWeek": 0.0,
            "ChristmasWeek": 0.0,
        }
    ]
}

# Envoyez une requête POST à l'URL du modèle
response = requests.post("http://0.0.0.0:5050/invocations", json=data_json_1)

# Affichez la prédiction

print(">>> Prédiction obtenue : %s " % response.json())

##########################################################
# PART 2 : inference with an inference pipeline          #
##########################################################
print("\n----- PARTIE 2 : prédiction avec un script d'inférence")
data_sales_2 = {
    "dataframe_records": [
        {
            "Store": 10,
            "Dept": 5,
            "Date": "2011-04-01",
            "IsHoliday": False,
        }
    ]
}
df_sales_2 = pd.DataFrame(data_sales_2["dataframe_records"])


# Chemin vers le run MLflow
experiment_id = "141226188525011380" # L'experiment id de votre modèle.
mlflow_run_id = "261565cbd9e34a3d9038cba159434dee"  # Mlflow run id de votre modèle.
artifact_uri = f"mlruns/{experiment_id}/{mlflow_run_id}/artifacts"


# 4.1.D : Chargez tous les artefacts
# ------------------------------------------------------------------------------------
df_features = pd.read_csv(f"{artifact_uri}/features.csv")
df_stores = pd.read_csv(f"{artifact_uri}/stores.csv")
features_transformer = joblib.load(f"{artifact_uri}/features_transformer.pkl")
cleaner = joblib.load(f"{artifact_uri}/cleaner.pkl")
data_collector = joblib.load(f"{artifact_uri}/data_collector.pkl")
# ------------------------------------------------------------------------------------

# 4.1.D : Ecrire la fonction d'application des artefacts (prepare_and_transform_data)
def prepare_and_transform_data(df_sales= pd.DataFrame()) -> pd.DataFrame:
    df_sales = data_collector.merge_sales_and_features(df_sales, df_features)
    df_sales = data_collector.merge_sales_and_stores(df_sales, df_stores)
    cleaned_data = cleaner.transform(df_sales)
    transformed_data = features_transformer.transform(cleaned_data)
    return transformed_data

# ------------------------------------------------------------------------------------

# 4.1.D : appliquer la fonction d'application des artefacts (prepare_and_transform_data)
# ------------------------------------------------------------------------------------
df_sales_2 = prepare_and_transform_data(df_sales_2)
# ------------------------------------------------------------------------------------


# Envoyez une requête POST à l'URL du modèle
data_json_2 = {"dataframe_records": df_sales_2.astype(str).to_dict(orient="records")}
response = requests.post("http://0.0.0.0:5050/invocations", json=data_json_2)

# Affichez la prédiction
print("Prédiction obtenue : %s " % response.json())

##########################################################
# PART 3 : alerting et monitoring                        #
##########################################################

sys.exit(0)  # TODO 4.2 : supprimez cette ligne


def predict_with_monitoring(data_json_to_predict: dict):
    """
    Predicts a given observations, logs the latency and the error
    :param data_json_to_predict: (dict) already processed.
    :return: (float) prediction
    """
    # prepare data and log latency
    start_time = time.time()
    try:
        prediction = requests.post(
            "http://0.0.0.0:5050/invocations", json=data_json_to_predict
        ).json()["predictions"][0]
        is_error = 0
    except:
        prediction = 0
        is_error = 1
    latency = time.time() - start_time
    is_very_high_value = 1 if prediction >= 50000 else 0
    log_response_info(latency, is_error, is_very_high_value)

    return prediction


def log_response_info(latency: float, is_error: float, is_very_high_value: float):
    """
    Logs the metrics latency, is_error and is_very_high_value is MLFlow.
    """
    # TODO 4.2 : logguez les métriques dans MLFLow
    # ------------------------------------------------------------------------------------
    #
    #
    # ------------------------------------------------------------------------------------


# Définissez cette expérimentation comme active
mlflow.set_experiment("4.2 monitoring")
with mlflow.start_run() as run:
    # on charge le jeu de test
    print("\n----- PARTIE 3 : monitoring and alerting")
    df_test = pd.read_csv(DATA_RAW / "test.csv")

    # on envoie chaque observation tous les 0.1 secondes
    for i in range(100):
        time.sleep(0.1)
        current_instance = pd.DataFrame(df_test.iloc[i]).T
        df_processed = prepare_and_transform_data(
            current_instance, df_features, df_stores
        )
        data_json_to_predict = {
            "dataframe_records": df_processed.to_dict(orient="records")
        }
        prediction = predict_with_monitoring(data_json_to_predict)
        print("live inference (%s) : prediction = %s" % (i, prediction))
