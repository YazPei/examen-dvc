import pandas as pd
from sklearn.model_selection import train_test_split
import os
import click

@click.command()
@click.option("--input", prompt="Chemin vers le fichier raw", help="Chemin du fichier CSV d'entrée.")
@click.option("--output", prompt="Dossier de sortie", help="Répertoire de sauvegarde des fichiers découpés.")
@click.option("--test_size", default=0.2, show_default=True, help="Proportion de test.")
@click.option("--random_state", default=42, show_default=True, help="Seed de reproductibilité.")
def split_data(input, output, test_size, random_state):
    """Découpe le dataset en jeux d'entraînement et de test, et les sauvegarde."""
    
    # Chargement
    df = pd.read_csv(input)

    # Séparation X / y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Création du dossier si besoin
    os.makedirs(output, exist_ok=True)

    # Sauvegarde
    X_train.to_csv(os.path.join(output, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output, "y_test.csv"), index=False)

    click.secho(f"Données sauvegardées dans : {output}", fg="green")

if __name__ == "__main__":
    split_data()

