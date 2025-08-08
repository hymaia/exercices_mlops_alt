import pandas as pd
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

from config import DATA_PROCESSED, REPORTS_DIR

if __name__ == "__main__":
    # Exercice 4.3.A : créer les jeux de référence et current
    # ------------------------------------------------------------------------------------
    x_train_processed = pd.read_parquet(
        DATA_PROCESSED / "x_train_processed.parquet"
    ).set_index("Date")
    x_val_processed = pd.read_parquet(
        DATA_PROCESSED / "x_val_processed.parquet"
    ).set_index("Date")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet")
    y_val = pd.read_parquet(DATA_PROCESSED / "y_val.parquet")
    pred_train = pd.read_parquet(DATA_PROCESSED / "pred_train.parquet")
    pred_val = pd.read_parquet(DATA_PROCESSED / "pred_val.parquet")

    y_train.index = x_train_processed.index
    y_val.index = x_val_processed.index

    pred_train.index = x_train_processed.index
    pred_val.index = x_val_processed.index

    df_reference = pd.concat([x_train_processed, y_train, pred_train], axis=1)
    df_current = pd.concat([x_val_processed, y_val, pred_val], axis=1)
    # ------------------------------------------------------------------------------------

    # Exercice 4.3.B : réaliser le ColumnMapping
    # ------------------------------------------------------------------------------------
    target = "Weekly_Sales"
    prediction = "prediction"
    date_col = "Date"
    numerical_features = [
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "Size",
        "MarkdownsSum",
        "Days_to_Thansksgiving",
        "Days_to_Christmas",
    ]

    categorical_features = [
        "Store",
        "Dept",
        "IsHoliday",
        "Type",
        "Day",
        "Week",
        "Month",
        "Year",
        "SuperBowlWeek",
        "LaborDayWeek",
        "ThanksgivingWeek",
        "ChristmasWeek",
    ]

    data_def = DataDefinition(
        numerical_columns=numerical_features,
        categorical_columns=categorical_features,
        regression=[Regression(target=target, prediction=prediction)],
    )

    # Convert pandas DataFrames to Evidently Dataset objects
    dataset_reference = Dataset.from_pandas(df_reference, data_definition=data_def)
    dataset_current = Dataset.from_pandas(df_current, data_definition=data_def)
    dataset_current_sample = Dataset.from_pandas(
        df_current.sample(10_000, random_state=42), data_definition=data_def
    )

    # ------------------------------------------------------------------------------------
    # créer un rapport de Model Drift
    model_drift_report = Report(
        metrics=[
            RegressionPreset(),
        ],
    )
    eval_model_drift = model_drift_report.run(dataset_current_sample, reference_data=None)
    eval_model_drift.save_html(str(REPORTS_DIR / "model_drift_report.html"))

    # Créer un rapport de Data Drift
    data_drift_report = Report(metrics=[DataDriftPreset()])
    eval_data_drift = data_drift_report.run(dataset_current, dataset_reference)
    eval_data_drift.save_html(str(REPORTS_DIR / "data_drift_report.html"))