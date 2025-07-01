import pandas as pd
import os
import joblib
import click
from sklearn.linear_model import Ridge # petit dataset, eviter overfitting
from sklearn.model_selection import GridSearchCV

@click.command()
@click.option('--input_features', prompt="Chemin vers X_train_scaled.csv", help="Chemin du fichier des features normalisés")
@click.option('--input_target', prompt="Chemin vers y_train.csv", help="Chemin du fichier des labels")
@click.option('--output_path', prompt="Chemin de sortie", help="Fichier .pkl pour sauvegarder les meilleurs paramètres")
def gridsearch(input_features, input_target, output_path):
    """Recherche des meilleurs hyperparamètres avec GridSearchCV"""

    # Chargement des données
    X_train = pd.read_csv(input_features)
    y_train = pd.read_csv(input_target).squeeze() 
    # Modèle de base
    model = Ridge()

    # Grille d’hyperparamètres à tester
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }

    # GridSearch
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Création dossier sortie
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarde des meilleurs paramètres
    joblib.dump(grid_search.best_params_, output_path)

    click.secho(f"Meilleurs paramètres sauvegardés dans : {output_path}")

if __name__ == "__main__":
    gridsearch()

