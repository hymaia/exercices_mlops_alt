import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

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

    column_mapping = ColumnMapping(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target=target,
        prediction=prediction,
    )

    # Prepare data for evidently
    df_reference_subset = df_reference
    df_current_subset = df_current
    df_current_sample = df_current.sample(10_000, random_state=42)

    # ------------------------------------------------------------------------------------
    # créer un rapport de Model Drift
    model_drift_report = Report(
        metrics=[
            RegressionPreset(),
        ],
    )
    model_drift_report.run(current_data=df_current_sample, reference_data=None, column_mapping=column_mapping)
    model_drift_report.save_html(str(REPORTS_DIR / "model_drift_report.html"))

    # Créer un rapport de Data Drift
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(current_data=df_current_subset, reference_data=df_reference_subset, column_mapping=column_mapping)
    data_drift_report.save_html(str(REPORTS_DIR / "data_drift_report.html"))