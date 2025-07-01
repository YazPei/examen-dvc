import pandas as pd
import pickle
import os
import click
from sklearn.linear_model import Ridge

@click.command()
@click.option('--input_features', prompt="Chemin vers X_train_scaled.csv")
@click.option('--input_target', prompt="Chemin vers y_train.csv")
@click.option('--best_params', prompt="Chemin vers best_params.pickle")
@click.option('--output_model', prompt="Chemin de sortie du modèle entraîné (.pkl)")
def train_model(input_features, input_target, best_params, output_model):
    """Entraîne un modèle Ridge et le sauvegarde."""

    # Chargement des données
    X_train = pd.read_csv(input_features)
    y_train = pd.read_csv(input_target).squeeze()

    # Chargement des meilleurs paramètres depuis un fichier pickle
    with open(best_params, "rb") as f:
        params = pickle.load(f)

    # Entraînement du modèle Ridge
    model = Ridge(**params)
    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    with open(output_model, "wb") as f:
        pickle.dump(model, f)

    click.secho(f"Modèle Ridge sauvegardé avec succès dans : {output_model}")

if __name__ == "__main__":
    train_model()

