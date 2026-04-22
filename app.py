import plotly.express as px
from numpy.linalg import svd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import shapiro
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden.*")
warnings.filterwarnings(
    "ignore", message="The library used by the .*country names.*")

numeric_cols = ['Score', 'GDP per capita', 'Social support',
                'Healthy life expectancy', 'Freedom to make life choices',
                'Generosity', 'Perceptions of corruption']

feature_cols = ['GDP per capita', 'Social support',
                'Healthy life expectancy', 'Freedom to make life choices',
                'Generosity', 'Perceptions of corruption']

expected_cols = ['Overall rank', 'Country or region'] + numeric_cols

# ================================
# VARIABELFÖRKLARINGAR
# ================================

# Källor:
# - Kaggle: https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2019.csv
# - World Happiness Report 2019, Statistical Appendix 1:
#   https://files.worldhappiness.report/WHR19_Statistical_Appendix_01.pdf
# - World Happiness Report 2019, Technical Box 1:
#   https://worldhappiness.report/ed/2019/changing-world-happiness/
#
# Kolumnnamnen kommer från 2019.csv. Definitionerna nedan bygger på WHR 2019
# Statistical Appendix 1, "Data Sources and Variable Definitions".
# Engelska källfraser behålls för att formuleringarna ska ligga nära originalet.
#
# Viktigt: featurekolumnerna i 2019.csv är inte råvärden. De är de sex
# faktorbidrag som WHR använder för att förklara skillnader i Score relativt
# Dystopia. WHR beskriver detta som "components explained by six hypothesized
# underlying determinants" och betonar att beräkningarna är "illustrative
# rather than conclusive".

variable_explanations = {
    'Overall rank': 'Rankning i 2019.csv; WHR beskriver detta som "ranking of happiness scores".',
    'Country or region': 'Land/region i 2019.csv; WHR använder även formuleringen "country/territory".',
    'Score': 'WHR: nationellt medelvärde för Cantril life ladder, med steg från 0 till 10.',
    'GDP per capita': 'Grundvariabel: GDP per capita i PPP. I 2019.csv visas faktorbidraget från GDP per capita.',
    'Social support': 'Grundvariabel: binärt GWP-svar på om man har släkt/vänner att räkna med vid problem.',
    'Healthy life expectancy': 'Grundvariabel: "Healthy life expectancies at birth" från WHO Global Health Observatory.',
    'Freedom to make life choices': 'Grundvariabel: GWP-frågan om man är nöjd/missnöjd med friheten att välja vad man gör med sitt liv.',
    'Generosity': 'Grundvariabel: residual från donationer senaste månaden efter regression mot GDP per capita.',
    'Perceptions of corruption': 'Grundvariabel: medel av två GWP-frågor: "Is corruption widespread throughout the government or not?" och "Is corruption widespread within businesses or not?" Vid saknad government-data används business corruption. I 2019.csv visas faktorbidraget, vilket motsvarar freedom from corruption/absence of corruption i WHR:s modell.',
}


# ================================
# DATAINTEGRITETSKONTROLL
# ================================

def check_data_integrity(df):
    print("\n--- Dataintegritetskontroll ---")
    problems = []

    # 1. Kontrollera saknade värden
    null_counts = df.isnull().sum()
    if null_counts.any():
        for col, count in null_counts[null_counts > 0].items():
            problems.append(f"Saknade värden: {count} stycken i '{col}'.")

    # 2. Kontrollera dubbletter
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        problems.append(f"{duplicates} dubbletter hittades i datasetet.")

    # 3. Kontrollera att förväntade kolumner finns.
    for col in expected_cols:
        if col not in df.columns:
            problems.append(f"Kolumnen '{col}' saknas i datasetet.")

    # 4. Kontrollera datatyper och rimlighet.
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Kolumnen '{col}' är inte numerisk.")
        elif (df[col] < 0).any():
            problems.append(f"Negativa värden hittades i '{col}'.")

    if 'Score' in df.columns and pd.api.types.is_numeric_dtype(df['Score']):
        if not df['Score'].between(0, 10).all():
            problems.append(
                "Score innehåller värden utanför Cantril-skalan 0-10.")

    if 'Overall rank' in df.columns and df['Overall rank'].duplicated().any():
        problems.append("Overall rank innehåller dubbletter.")

    if 'Country or region' in df.columns and df['Country or region'].duplicated().any():
        problems.append("Country or region innehåller dubbletter.")

    if problems:
        print("Varning: Följande problem identifierades:")
        for p in problems:
            print(f"- {p}")
    else:
        print("✓ Datan är intakt och redo för analys (inga saknade värden, dubbletter eller negativa värden).")
    print("-------------------------------\n")
    return not problems


# Score och rankningen bygger på data från Gallup World Poll.
# Score baseras på Cantril-stegen, där respondenter skattar sitt liv
# på en skala från 0 (sämsta tänkbara liv) till 10 (bästa tänkbara liv).

# ================================
# LÄS IN OCH KONTROLLERA DATA
# ================================

# Läs in datasetet.
df = pd.read_csv('2019.csv')

# Kör integritetskontrollen.
if not check_data_integrity(df):
    raise SystemExit(
        "Analysen avbryts eftersom datasetet inte klarade integritetskontrollen.")

# Överblick över kolumner, datatyper och antal saknade värden.
df.info()


# ================================
# SHAPIRO-TEST
# ================================

# Testa om Score kan betraktas som normalfördelad.
def hypotes_test(x: float):
    if x >= 0.05:
        svar = "Nollhypotesen förkastas ej."
    else:
        svar = "Nollhypotesen förkastas."
    return svar


test = shapiro(df["Score"])
svar = hypotes_test(test.pvalue)

print(
    f"\nShapiro (H0: Score är normalfördelad) - \nteststatistik: {test.statistic:.3f}. p-värde: {test.pvalue:.3f}\nResultat: {svar}")


# ================================
# VISUALISERING AV SCORE
# ================================

# Visa Score efter Overall rank.
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Overall rank', y='Score',
                s=55, color='#2f6f9f', edgecolor='white')
plt.title('Fördelning av Score efter Overall rank', fontsize=14)
plt.xlabel('Overall rank', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ================================
# KARTA
# ================================

# Visualisera Score per Country or region på en världskarta.
fig = px.choropleth(
    df,
    locations="Country or region",
    locationmode="country names",
    color="Score",
    color_continuous_scale="RdYlGn",
    title="Global fördelning av Score"
)

fig.show()


# ================================
# PERCEIVED LACK OF CORRUPTION, GDP PER CAPITA OCH SCORE
# ================================

# Undersök relationen mellan GDP per capita, Score och Perceived lack of corruption.
fig, ax = plt.subplots(figsize=(12, 7))

# Sätt GDP per capita på x-axeln, Score på y-axeln och låt färg och storlek visa Perceived lack of corruption.
# Färgskalan använder datasetets faktorbidragsskala och normaliseras inte till 0-1.
corruption_values = df['Perceptions of corruption']
point_sizes = 70 + (corruption_values * 120)
scatter_plot = ax.scatter(
    df['GDP per capita'],
    df['Score'],
    c=corruption_values,
    cmap='Greens',
    vmin=0,
    vmax=corruption_values.max(),
    s=point_sizes,
    alpha=0.8,
    edgecolors='black',
    linewidths=0.6
)

# Lägg till en enkel trendlinje mellan GDP per capita och Score.
trend_x = np.linspace(df['GDP per capita'].min(),
                      df['GDP per capita'].max(), 100)
trend_m, trend_b = np.polyfit(df['GDP per capita'], df['Score'], 1)
ax.plot(trend_x, trend_m * trend_x + trend_b, color='black', linestyle='--', linewidth=1.5,
        label='Linjär trend')

# Etikettera länder med högst och lägst Perceived lack of corruption.
label_df = pd.concat([
    df.nlargest(4, 'Perceptions of corruption'),
    df.nsmallest(4, 'Perceptions of corruption')
]).drop_duplicates(subset='Country or region')

for _, row in label_df.iterrows():
    ax.annotate(
        row['Country or region'],
        (row['GDP per capita'], row['Score']),
        textcoords='offset points',
        xytext=(6, 5),
        fontsize=9,
        fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5)
    )

# Lägg till tydlig titel, axelrubriker och färgskala.
ax.set_title(
    'Jämförelse mellan GDP per capita, Score och Perceived lack of corruption', fontsize=16, pad=20)
ax.set_xlabel('GDP per capita', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
colorbar = fig.colorbar(scatter_plot, ax=ax)
colorbar.set_label('Perceived lack of corruption (faktorbidrag)')
colorbar.set_ticks(np.linspace(0, corruption_values.max(), 5))
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# ================================
# SVD-ANALYS
# ================================

# Jämför SVD med enbart centrering och med standardisering.
print("\nSVD-analys:")
X = df[feature_cols].values

# 1. Centrera variablerna utan skalning.
X_centered = X - X.mean(axis=0)

U_c, S_c, Vt_c = svd(X_centered, full_matrices=False)
explained_var_c = (S_c**2 / np.sum(S_c**2)) * 100

# 2. Standardisera variablerna (centrering + skalning).
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
U_s, S_s, Vt_s = svd(X_std, full_matrices=False)
explained_var_s = (S_s**2 / np.sum(S_s**2)) * 100

# Skriv ut absoluta variabelbidrag med variabelnamnen som rader.
print("\nSVD: absoluta variabelbidrag (endast centrering - GDP per capita dominerar kraftigt)")
svd_df_c = pd.DataFrame(np.abs(Vt_c), columns=feature_cols, index=[
                        f"K{i+1}" for i in range(len(S_c))])
print(svd_df_c.iloc[:2].T.round(3))

print("\nSVD: absoluta variabelbidrag (standardiserad data - jämnare fördelning)")
svd_df_s = pd.DataFrame(np.abs(Vt_s), columns=feature_cols, index=[
                        f"K{i+1}" for i in range(len(S_s))])
print(svd_df_s.iloc[:2].T.round(3))

# Visualisera skillnaden i förklarad varians.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Visa vilken variabel som bidrar mest till varje komponent i diagrammen.
top_vars_c = svd_df_c.idxmax(axis=1)
top_vars_s = svd_df_s.idxmax(axis=1)
top_text_c = "Största bidrag:\n" + \
    "\n".join([f"{k}: {v}" for k, v in top_vars_c.items()])
top_text_s = "Största bidrag:\n" + \
    "\n".join([f"{k}: {v}" for k, v in top_vars_s.items()])

ax1.bar([f"K{i+1}" for i in range(len(S_c))], explained_var_c, color='skyblue')
ax1.set_title("Förklarad varians (endast centrering)")
ax1.set_ylabel("Förklarad varians (%)")
for i, v in enumerate(explained_var_c):
    ax1.text(i, v + 1, f"{v:.1f}%", ha='center')
ax1.text(0.98, 0.95, top_text_c, transform=ax1.transAxes, va='top', ha='right',
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85))

ax2.bar([f"K{i+1}" for i in range(len(S_s))], explained_var_s, color='salmon')
ax2.set_title("Förklarad varians (standardiserad data)")
for i, v in enumerate(explained_var_s):
    ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
ax2.text(0.98, 0.95, top_text_s, transform=ax2.transAxes, va='top', ha='right',
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85))

plt.tight_layout()
plt.show()

# ================================
# UMAP
# ================================

print("\nJämförelse av skalningsmetoder inför UMAP:")
print("- Utan skalning: känslig för variabler med stora värden (t.ex. GDP per capita)")
print("- Standardisering: balanserar variabler och ger ofta tydligare struktur")
print("- Normalisering: kan komprimera variation och ibland ge sämre separation\n")

# Funktion för att visualisera UMAP med Plotly.


def plot_umap(umap_data, title):
    fig = px.scatter(
        umap_data,
        x="UMAP1",
        y="UMAP2",
        color="Score",
        hover_name="Country or region",
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


# Använd faktorvärdena direkt från 2019.csv, utan extra skalning.
X_raw = df[feature_cols]

# Skapa en UMAP-reducerare.
reducer_raw = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=200
)

# Anpassa modellen och transformera datan.
embedding_raw = reducer_raw.fit_transform(X_raw)

# Skapa en DataFrame för visualisering.
umap_df_raw = pd.DataFrame(embedding_raw, columns=["UMAP1", "UMAP2"])
umap_df_raw["Score"] = df["Score"]
umap_df_raw["Country or region"] = df["Country or region"]

# Plotta resultatet.
plot_umap(umap_df_raw, "UMAP-projektion (utan skalning)")


# ================================
# UMAP MED STANDARDISERING
# ================================

# Standardisera variablerna (medelvärde = 0, standardavvikelse = 1).
# z = (x - medelvärdet av x) / standardavvikelse.
# Värdena centreras runt 0 men är inte begränsade till intervallet [-1, 1].
standard_scaler = StandardScaler()
X_standardised = standard_scaler.fit_transform(df[feature_cols])

# Anpassa UMAP på standardiserad data.
reducer_std = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=200
)

embedding_std = reducer_std.fit_transform(X_standardised)

# Skapa en DataFrame för visualisering.
umap_df_std = pd.DataFrame(embedding_std, columns=["UMAP1", "UMAP2"])
umap_df_std["Score"] = df["Score"]
umap_df_std["Country or region"] = df["Country or region"]

# Visualisera resultatet.
plot_umap(umap_df_std, "UMAP-projektion (standardiserad data)")

# Jämför grupperingen med originalvariablerna innan resultatet tolkas.


# ================================
# UMAP MED MIN-MAX-NORMALISERING
# ================================

# Normalisera variablerna till intervallet [0, 1].
# Minsta värdet blir 0 och största värdet blir 1 för varje variabel.
minmax_scaler = MinMaxScaler()
X_normalised = minmax_scaler.fit_transform(df[feature_cols])

# Anpassa UMAP på normaliserad data.
reducer_norm = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=200
)

embedding_norm = reducer_norm.fit_transform(X_normalised)

# Skapa en DataFrame för visualisering.
umap_df_norm = pd.DataFrame(embedding_norm, columns=["UMAP1", "UMAP2"])
umap_df_norm["Score"] = df["Score"]
umap_df_norm["Country or region"] = df["Country or region"]

# Visualisera resultatet.
plot_umap(umap_df_norm, "UMAP-projektion (min-max-normaliserad data)")

# Jämför grupperingen med den standardiserade versionen innan resultatet tolkas.
