# Importar librerías
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Descargar dataset
path = kagglehub.dataset_download("wardabilal/spotify-global-music-dataset-20092025")
print("Archivos descargados en:", path)

# Listar archivos disponibles
print("Contenido de la carpeta:")
for f in os.listdir(path):
    print(f)

# Cargar dataset principal
file_path = os.path.join(path, "track_data_final.csv")
df = pd.read_csv(file_path)

print("Dimensiones:", df.shape)
print("Columnas:", df.columns)
print("Primeras filas:\n", df.head())

# Revisar valores nulos
print(df.isnull().sum())

# Convertir fechas y extraer año
df['album_release_date'] = pd.to_datetime(df['album_release_date'], errors='coerce')
df['release_year'] = df['album_release_date'].dt.year
print(df['release_year'].min(), df['release_year'].max())

# Popularidad promedio por año
popularity_per_year = df.groupby('release_year')['track_popularity'].mean()
plt.figure(figsize=(12,6))
sns.lineplot(x=popularity_per_year.index, y=popularity_per_year.values, marker="o")
plt.title("Popularidad promedio de canciones por año")
plt.xlabel("Año de lanzamiento")
plt.ylabel("Popularidad promedio")
plt.grid(True)
plt.show()

# Cantidad de canciones por año
songs_per_year = df['release_year'].value_counts().sort_index()
plt.figure(figsize=(12,6))
sns.barplot(x=songs_per_year.index, y=songs_per_year.values, palette="mako")
plt.title("Cantidad de canciones por año")
plt.xlabel("Año de lanzamiento")
plt.ylabel("Número de canciones")
plt.xticks(rotation=45)
plt.show()

# Top 10 artistas por popularidad promedio
top_artists = (
    df.groupby('artist_name')['track_popularity']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
print("Top 10 artistas por popularidad promedio:")
print(top_artists)

plt.figure(figsize=(12,6))
sns.barplot(x=top_artists.values, y=top_artists.index, palette="crest")
plt.title("Top 10 artistas por popularidad promedio")
plt.xlabel("Popularidad promedio")
plt.ylabel("Artista")
plt.show()

# Top 10 géneros más frecuentes
top_genres = df['artist_genres'].value_counts().head(10)
print("Top 10 géneros más frecuentes:")
print(top_genres)

plt.figure(figsize=(12,6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette="rocket")
plt.title("Top 10 géneros musicales más frecuentes")
plt.xlabel("Número de canciones")
plt.ylabel("Género")
plt.show()

# Duración promedio por año (en minutos)
duration_per_year = df.groupby('release_year')['track_duration_ms'].mean() / 60000
plt.figure(figsize=(12,6))
sns.lineplot(x=duration_per_year.index, y=duration_per_year.values, marker="o", color="orange")
plt.title("Duración promedio de canciones por año")
plt.xlabel("Año de lanzamiento")
plt.ylabel("Duración promedio (minutos)")
plt.grid(True)
plt.show()

# Relación entre duración y popularidad
df['duration_min'] = df['track_duration_ms'] / 60000
plt.figure(figsize=(12,6))
sns.scatterplot(x=df['duration_min'], y=df['track_popularity'], alpha=0.5, color="purple")
plt.title("Relación entre duración y popularidad de canciones")
plt.xlabel("Duración (minutos)")
plt.ylabel("Popularidad")
plt.grid(True)
plt.show()

# 🔎 Matriz de correlación entre variables clave
corr = df[['track_popularity','duration_min','artist_popularity','artist_followers']].corr()
print("Matriz de correlación:\n", corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación entre popularidad, duración y métricas de artista")
plt.show()














