import pandas as pd
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

from config import REPORTS_DIR

if __name__ == "__main__":

    # TODO - exercice 4.3.A : créer les jeux de référence et current
    # ------------------------------------------------------------------------------------
    #
    #
    #
    #
    #
    #
    #
    # ------------------------------------------------------------------------------------

    # TODO - exercice 4.3.B : réaliser le ColumnMapping
    # ------------------------------------------------------------------------------------
    # target =
    # prediction =
    #  numerical_features = []
    # categorical_features = []
    # data_def = DataDefinition(
    #     numerical_columns=numerical_features,
    #     categorical_columns=categorical_features,
    #     regression=[Regression(target=target, prediction=prediction)],
    # )
    # dataset_reference = 
    # dataset_current =
    # dataset_current_sample = 
    #
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
