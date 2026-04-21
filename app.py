import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from scipy.stats import shapiro
import statsmodels.api as sm
import numpy as np
from numpy.linalg import svd
import plotly.express as px

# The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale.

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

# Visualisera på karta
fig = px.choropleth(
    df,
    locations="Country or region",
    locationmode="country names",
    color="Score",
    color_continuous_scale="RdYlGn",
    title="Global Happiness Score"
)

fig.show()

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

# Visualisera umap
umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
umap_df["Score"] = df["Score"]
umap_df["Country"] = df["Country or region"]

fig = px.scatter(
    umap_df,
    x="UMAP1",
    y="UMAP2",
    color="Score",
    hover_name="Country",
    color_continuous_scale="Viridis",
    title="UMAP Projection Colored by Happiness"
)

fig.update_traces(
    marker=dict(
        size=9,
        opacity=0.85,
        line=dict(width=1, color="black")
    )
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="#f7f7f7"
)

fig.show()