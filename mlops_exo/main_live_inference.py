import pandas as pd
import requests
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


##########################################################
# PART 1 : inference with all features already computes  #
##########################################################

print("----- PARTIE 1 : prédiction avec toutes les features déjà données")
data_json_1 = {
    "dataframe_records": [{"Store": 8986.8302954228,
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
                           "ChristmasWeek": 0.0
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
    "dataframe_records": [{
                        "Store": 10,
                        "Dept": 5,
                        "Date": "2011-04-01"
                         }]
}


# Chemin vers le run MLflow
mlflow_run_id = "6e9bbd60083546f48027c00eb038a4ac"
artifact_uri = f"mlruns/0/{mlflow_run_id}/artifacts"


# Load cleaner and transformer and apply it on data
cleaner = joblib.load(os.path.join(artifact_uri, "cleaner.pkl"))
features_transformer = joblib.load(os.path.join(artifact_uri, "features_transformer.pkl"))
df_features = pd.read_csv(os.path.join(artifact_uri, "features.csv"))
df_stores = pd.read_csv(os.path.join(artifact_uri, "stores.csv"))
df_sales_2 = pd.DataFrame(data_sales_2["dataframe_records"])
df_sales_2 = df_sales_2.merge(df_features, on=["Store", "Date"])
df_sales_2 = df_sales_2.merge(df_stores, on=["Store"])
df_sales_2 = cleaner.transform(df_sales_2)
df_sales_2 = features_transformer.transform(df_sales_2)
data_json_2 = {"dataframe_records": df_sales_2.to_dict(orient="records")}

# apply transformations
# Envoyez une requête POST à l'URL du modèle
response = requests.post("http://0.0.0.0:5050/invocations", json=data_json_2)

# Affichez la prédiction
print("Prédiction obtenue : %s " % response.json())

##########################################################
# PART 3 : alerting et monitoring                        #
##########################################################
