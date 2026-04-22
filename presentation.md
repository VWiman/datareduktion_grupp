# Argumentation för val av Datareduktionsmetod

## Vald metod: UMAP (Uniform Manifold Approximation and Projection)

Vi har valt att använda UMAP för att analysera och visualisera grupper av observationer i vårt dataset (2019.csv). Argumenten nedan är rangordnade efter betydelse för vårt val:

### 1. Förmåga att identifiera tydliga kluster
Detta är det viktigaste argumentet då vårt primära mål är att se om länderna bildar tydliga grupper. UMAP är "state-of-the-art" på att "dra ihop" liknande punkter och "trycka isär" olika punkter, vilket skapar visuella kluster som är betydligt lättare att tolka än i t.ex. PCA.

### 2. Icke-linjära samband
Sociala och ekonomiska faktorer (som lycka, korruption och generositet) har sällan enkla, linjära relationer. UMAP:s största tekniska fördel är att den kan fånga upp dessa komplexa mönster som en linjär metod som PCA riskerar att missa eller "platta till".

### 3. Bevarande av både lokal och global struktur
För att vår gruppering ska vara trovärdig måste vi veta både vilka länder som liknar varandra mest (lokalt) och hur de stora grupperna förhåller sig till varandra (globalt). UMAP balanserar detta på ett sätt som ger en mer rättvisande helhetsbild än andra metoder.

Varför inte andra metoder?
- **PCA (Principal Component Analysis):** Bra för att hitta största variansen, men sämre på att separera kluster i komplex data (Slide 11).
- **Korrespondensanalys (CA):** Enbart för kategorisk data (Slide 10). Vår data är kontinuerlig och numerisk, vilket gör metoden tekniskt oanvändbar här.

## Slutsats
UMAP är det mest lämpade verktyget för att uppfylla vårt mål: att genom en kraftfull visualisering identifiera och kommunicera hur världens länder grupperar sig baserat på lyckoprofiler.

