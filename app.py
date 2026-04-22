import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from scipy.stats import shapiro
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from numpy.linalg import svd
import plotly.express as px

# The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale.

# Ladda datasetet.
df = pd.read_csv('2019.csv')

# Överblick över kolumner, datatyper och antal saknade värden.
df.info()

# Vi gör ett shapiro test på Score


def hypotes_test(x: float):
    if x >= 0.05:
        svar = "Nollhypotesen förkastas ej."
    else:
        svar = "Nollhypotesen förkastas."
    return svar


test = shapiro(df["Score"])
svar = hypotes_test(test.pvalue)

print(
    f"\nShapiro (Ho: Normalfördelning) - \nstatistik: {test.statistic:.3f}. p-value: {test.pvalue:.3f}\nResultat: {svar}")

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
    print(
        f"{col}: σ = {S[i]:.4f}, förklarad varians = {(S[i]**2 / np.sum(S**2))*100:.2f}%")

svd_df = pd.DataFrame(Vt, columns=svd_cols, index=[
                      f"Komponent {i+1}" for i in range(len(S))])
print("\nVariabelbidrag per komponent:")
print(svd_df.round(3))

# ================================
# UMAP
# ================================

feature_cols = ['GDP per capita', 'Social support',
                'Healthy life expectancy', 'Freedom to make life choices',
                'Generosity', 'Perceptions of corruption']

print("\nJämförelse av skalningsmetoder inför UMAP:")
print("- Utan skalning: känslig för variabler med stora värden (t.ex. BNP)")
print("- Standardisering: balanserar variabler → ofta bäst struktur")
print("- Normalisering: kan komprimera variation → ibland sämre separation\n")

# Funktion för att visualisera UMAP med Plot


def plot_umap(umap_data, title):
    fig = px.scatter(
        umap_data,
        x="UMAP1",
        y="UMAP2",
        color="Score",
        hover_name="Country",
        color_continuous_scale="Viridis",
        title=title
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.85, line=dict(width=1, color="black"))
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="#f7f7f7"
    )
    fig.show()

# ================================
# UMAP UTAN SKALNING
# ================================


# Använd råa feature-värden (ingen skalning appliceras)
X_raw = df[feature_cols]

# Skapa UMAP-reducer
reducer_raw = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=200
)

# Anpassa modellen och transformera datan
embedding_raw = reducer_raw.fit_transform(X_raw)

# Skapa DataFrame för visualisering
umap_df_raw = pd.DataFrame(embedding_raw, columns=["UMAP1", "UMAP2"])
umap_df_raw["Score"] = df["Score"]
umap_df_raw["Country"] = df["Country or region"]

# Plotta resultatet
plot_umap(umap_df_raw, "UMAP-projektion (utan skalning)")


# ================================
# UMAP MED STANDARDISERING
# ================================

# Standardisera variabler (medelvärde = 0, standardavvikelse = 1)
# z = (x - medelvärdet av x) / standardavvikelse
# -1.00 till 1.00
standard_scaler = StandardScaler()
X_standardised = standard_scaler.fit_transform(df[feature_cols])

# Anpassa UMAP på standardiserad data
reducer_std = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=200
)

embedding_std = reducer_std.fit_transform(X_standardised)

# Skapa DataFrame
umap_df_std = pd.DataFrame(embedding_std, columns=["UMAP1", "UMAP2"])
umap_df_std["Score"] = df["Score"]
umap_df_std["Country"] = df["Country or region"]

# Visualisera resultatet
plot_umap(umap_df_std, "UMAP-projektion (standardiserad data)")

# Vi kan se ett att grupperna verkar vara grupperade utifrån förväntad livslängde, gdp och korruption.


# ================================
# UMAP MED MIN-MAX-NORMALISERING
# ================================

# Normalisera variabler till intervallet [0, 1]
# 0.00 till 1.00
minmax_scaler = MinMaxScaler()
X_normalised = minmax_scaler.fit_transform(df[feature_cols])

# Anpassa UMAP på normaliserad data
reducer_norm = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=200
)

embedding_norm = reducer_norm.fit_transform(X_normalised)

# Skapa DataFrame
umap_df_norm = pd.DataFrame(embedding_norm, columns=["UMAP1", "UMAP2"])
umap_df_norm["Score"] = df["Score"]
umap_df_norm["Country"] = df["Country or region"]

# Visualisera resultatet
plot_umap(umap_df_norm, "UMAP-projektion (min-max-normaliserad data)")

# Här ser vi att Freedom to make life choices verkar påverkar hur datan grupperas