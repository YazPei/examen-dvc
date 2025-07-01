import pandas as pd
import joblib
import os
import click
from sklearn.linear_model import Ridge

@click.command()
@click.option('--input_features', prompt="Chemin vers X_train_scaled.csv")
@click.option('--input_target', prompt="Chemin vers y_train.csv")
@click.option('--best_params', prompt="Chemin vers best_params.pkl")
@click.option('--output_model', prompt="Chemin de sortie du modèle entraîné")
def train_model(input_features, input_target, best_params, output_model):
    """Entraîne un modèle Ridge et le sauvegarde."""

    X_train = pd.read_csv(input_features)
    y_train = pd.read_csv(input_target).squeeze()
    params = joblib.load(best_params)

    model = Ridge(**params)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(model, output_model)

    click.secho(f"Modèle sauvegardé dans : {output_model}")

if __name__ == "__main__":
    train_model()

