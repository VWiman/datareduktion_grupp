import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from scipy.stats import shapiro
import statsmodels.api as sm
import numpy as np
from numpy.linalg import svd

# Ladda datasetet.
df = pd.read_csv('2019.csv')

# Överblick över kolumner, datatyper och antal saknade värden.
df.info()

# Vi gör ett shapiro test på Score
def hypotes_test(x:float):
    if x >= 0.05:
        svar="Nollhypotesen förkastas ej."
    else: svar = "Nollhypotesen förkastas."
    return svar

test = shapiro(df["Score"])
svar = hypotes_test(test.pvalue)

print(f"\nShapiro (Ho: Normalfördelning) - \nstatistik: {test.statistic:.3f}. p-value: {test.pvalue:.3f}\nResultat: {svar}")

# Visualisera datan med pairplot
numeric_cols = ['Score', 'GDP per capita', 'Social support',
                'Healthy life expectancy', 'Freedom to make life choices',
                'Generosity', 'Perceptions of corruption']

scatterplot_cols = ['Score']

plt.figure(figsize=(10, 8))
sns.scatterplot(df[scatterplot_cols], markers=True)
plt.title('Distribution of Happiness Score (Index Position)', fontsize=14)
plt.xlabel('Country Index (Rank Order)', fontsize=12)
plt.ylabel('Happiness Score Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

svd_cols = ['GDP per capita', 'Social support',
                'Healthy life expectancy', 'Freedom to make life choices',
                'Generosity', 'Perceptions of corruption']

# Centrera datan
X = df[svd_cols].values
X_centered = X - X.mean(axis=0)

# Beräkna SVD
U, S, Vt = svd(X_centered, full_matrices=False)

# Visa singulärvärdena per variabel/komponent
for i, col in enumerate(svd_cols):
    print(f"{col}: σ = {S[i]:.4f}, förklarad varians = {(S[i]**2 / np.sum(S**2))*100:.2f}%")

svd_df = pd.DataFrame(Vt, columns=svd_cols, index=[f"Komponent {i+1}" for i in range(len(S))])
print("\nVariabelbidrag per komponent:")
print(svd_df.round(3))

# Skapa en reducer med umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, verbose=True, target_metric = "categorical", n_jobs=1, random_state=200, metric="euclidean")
embedding = reducer.fit_transform(df[numeric_cols])

# Skapa figuren
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df["Score"], cmap="viridis", s=30)
plt.colorbar(scatter, label="Score")
plt.title("UMAP — länder")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()