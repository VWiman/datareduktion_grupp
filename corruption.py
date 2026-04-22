import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from scipy.stats import shapiro
import numpy as np

# Ladda datasetet
df = pd.read_csv('2019.csv')

df['Actual Corruption'] = 1 - df['Perceptions of corruption']
print(df)

def hypotes_test(p_val: float):
    if p_val >= 0.05:
        return "Nollhypotesen förkastas ej (Data ser normalfördelad ut)."
    else:
        return "Nollhypotesen förkastas (Data är inte normalfördelad)."

test = shapiro(df["Score"])
svar = hypotes_test(test.pvalue)
print(f"Shapiro Test Resultat: {svar}")

# NY GRAF: Jämförelse av Corruption, GDP och Happiness
plt.figure(figsize=(12, 7))

# Vi sätter GDP på X, Score på Y, och låter färgen (hue) och storleken visa Korruption
scatter_plot = sns.scatterplot(
    data=df, 
    x='GDP per capita', 
    y='Score', 
    hue='Actual Corruption', 
    size='Actual Corruption',
    palette='flare',
    sizes=(40, 400),
    alpha=0.7
)

# Lägg till tydliga titlar och etiketter
plt.title('Comparison of Wealth (GDP), Happiness (Score), and Corruption', fontsize=16, pad=20)
plt.xlabel('GDP per capita (Economic Strength)', fontsize=12)
plt.ylabel('Happiness Score (Index)', fontsize=12)

# Flytta förklaringen (legend) så den inte täcker grafen
plt.legend(title='Amount of Corruption', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# UMAP Dimensionsreduktion
numeric_cols = ['Score', 'GDP per capita', 'Social support',
                'Healthy life expectancy', 'Freedom to make life choices',
                'Generosity', 'Perceptions of corruption']

# Skapa reducer
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, verbose=False, n_jobs=1, random_state=200)
embedding = reducer.fit_transform(df[numeric_cols])

# Skapa UMAP-figuren
plt.figure(figsize=(10, 8))
plt.scatter(
    embedding[:, 0], 
    embedding[:, 1], 
    c=df['Score'],
    cmap='viridis', 
    s=50, 
    edgecolor='white', 
    linewidth=0.5
)

plt.title("UMAP: Clustering Countries by All Variables", fontsize=14)
plt.xlabel("UMAP Component 1", fontsize=12)
plt.ylabel("UMAP Component 2", fontsize=12)

cbar = plt.colorbar()
cbar.set_label('Happiness Score', rotation=270, labelpad=15)

plt.show()