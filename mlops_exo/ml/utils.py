

def print_mlflow_artefact_uri():
    """
    Get the MLflow run id from the environment variable
    :return: (str) MLflow run id
    """
    try:
        import mlflow
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Les artefacts sont sauvegardés ici : {artifact_uri}")
    except:
        print("pas de sauvegarde avec MLFlow actuellement")
