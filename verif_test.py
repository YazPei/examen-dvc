import pandas as pd

# Chargement du fichier CSV
df = pd.read_csv("/home/yazpei/my_exams/DVC/Examen_DVC/examen-dvc/data/raw_data/raw")

# Affichage des premières lignes
print(df.head())

# Info sur le dataset
print(df.info())
