import pandas as pd
import joblib
import json
import os
import click
from sklearn.metrics import r2_score, mean_squared_error

@click.command()
@click.option('--input_features', prompt="Chemin vers X_test_scaled.csv")
@click.option('--input_target', prompt="Chemin vers y_test.csv")
@click.option('--input_model', prompt="Chemin vers model.joblib")
@click.option('--output_predictions', prompt="Chemin vers predictions.csv")
@click.option('--output_metrics', prompt="Chemin vers scores.json")
def evaluate_model(input_features, input_target, input_model, output_predictions, output_metrics):
    """Évalue le modèle et enregistre les scores et les prédictions"""

    # Chargement des données
    X_test = pd.read_csv(input_features)
    y_test = pd.read_csv(input_target).squeeze()
    model = joblib.load(input_model)

    # Prédictions
    y_pred = model.predict(X_test)

    # Sauvegarde des prédictions
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    os.makedirs(os.path.dirname(output_predictions), exist_ok=True)
    pred_df.to_csv(output_predictions, index=False)

    # Calcul des scores
    scores = {
        "r2_score": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(scores, f, indent=4)

    click.secho(f"Prédictions : {output_predictions}")
    click.secho(f"Scores     : {output_metrics}")

if __name__ == "__main__":
    evaluate_model()
