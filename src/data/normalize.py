import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import click

@click.command()
@click.option('--input_train', prompt="Chemin vers X_train.csv", help="Chemin du fichier X_train")
@click.option('--input_test', prompt="Chemin vers X_test.csv", help="Chemin du fichier X_test")
@click.option('--output_dir', prompt="Dossier de sortie", help="Répertoire pour les fichiers normalisés")
def normalize_data(input_train, input_test, output_dir):
    """Normalise les données d'entraînement et de test"""

    # Chargement
    X_train = pd.read_csv(input_train)
    X_test = pd.read_csv(input_test)

    # Colonnes numériques uniquement
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_scaled = scaler.transform(X_test[num_cols])

    # DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=num_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=num_cols)

    # Dossier sortie
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarde
    X_train_scaled_df.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)

    click.secho(f"Données normalisées sauvegardées dans {output_dir}")

if __name__ == "__main__":
    normalize_data()

