# =========================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =========================================================
# ruff: noqa: E402
from sklearnex import patch_sklearn

patch_sklearn()  # small patch to make sklearn go faster on intel processors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configuração gráfica
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =========================================================
# 2. IMPORTAÇÃO DOS DADOS
# =========================================================
# Carregar os datasets de treino e teste
train_data = pd.read_csv("Datasets/train.csv")
test_data = pd.read_csv("Datasets/test.csv")

continuous_cols = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# =========================================================
# 3. DESCRIÇÃO E EXPLORAÇÃO INICIAL
# =========================================================
print("\nPrimeiras linhas:")
print(train_data.head())

print("\nDimensão do dataset:")
print(train_data.shape)

print("\nInformações gerais:")
print(train_data.info())

print("\nEstatísticas descritivas:")
print(train_data.describe().T)

# =========================================================
# 4. TRATAMENTO DE OUTLIERS (MÉTODO IQR)
# =========================================================
iqr_outliers_summary = {}
all_outlier_indices = set()

for col in continuous_cols:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identificar outliers
    outliers = train_data[
        (train_data[col] < lower_bound) | (train_data[col] > upper_bound)
    ]
    iqr_outliers_summary[col] = len(outliers)
    all_outlier_indices.update(outliers.index)

print("\nContagem de Outliers usando o Método IQR:")
for col, count in iqr_outliers_summary.items():
    print(f" - {col}: {count} outliers")
print(
    f"Total de linhas únicas contendo pelo menos um outlier (IQR): {len(all_outlier_indices)}"
)

# Guardar boxplots dos outliers
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    sns.boxplot(x=train_data[col], ax=axes[i], color="skyblue")
    axes[i].set_title(f"Boxplot de {col}")

plt.tight_layout()
plt.savefig("Graphs/outliers_boxplots.png")
plt.close()

# =========================================================
# 5. DISTRIBUIÇÃO DA VARIÁVEL TARGET
# =========================================================
satisfaction_counts = (
    train_data["satisfaction"].value_counts().sort_values(ascending=False)
)

sns.barplot(
    x=satisfaction_counts.index,
    y=satisfaction_counts.values,
    hue=satisfaction_counts.index,
    palette="viridis",
    legend=False,
)
plt.title("Distribuição da Variável Target: Satisfação")
plt.xlabel("Nível de Satisfação")
plt.ylabel("Contagem")
plt.tight_layout()
plt.savefig("Graphs/satisfaction_distribution.png")
plt.close()


# =========================================================
# 6. FUNÇÃO DE LIMPEZA E PREPARAÇÃO (TREINO E TESTE)
# =========================================================
def preprocess_pipeline(df):
    # 1. Remover identificadores irrelevantes
    df_clean = df.drop(columns=["Unnamed: 0", "id"], errors="ignore")

    # 2. Tratar valores em falta na coluna de chegada com base na partida
    df_clean["Arrival Delay in Minutes"] = df_clean["Arrival Delay in Minutes"].fillna(
        df_clean["Departure Delay in Minutes"]
    )

    # 3. Aplicar Transformação Logarítmica para reduzir a assimetria (skewness)
    df_clean["Log_Departure_Delay"] = np.log1p(df_clean["Departure Delay in Minutes"])
    df_clean["Log_Arrival_Delay"] = np.log1p(df_clean["Arrival Delay in Minutes"])

    # 4. Criar variável binária de presença de atraso
    df_clean["Is_Delayed"] = (df_clean["Departure Delay in Minutes"] > 0).astype(int)

    # 5. Mapeamento Ordinal da coluna Class
    df_clean["Class"] = df_clean["Class"].map(
        {"Eco": 0, "Eco Plus": 1, "Business": 2}.get
    )

    # 6. One-Hot Encoding para as restantes variáveis categóricas nominais
    df_clean = pd.get_dummies(
        df_clean, columns=["Gender", "Customer Type", "Type of Travel"], drop_first=True
    )

    # 7. Mapear a variável Target 'satisfaction' para valores binários (0 e 1)
    df_clean["satisfaction"] = df_clean["satisfaction"].map(
        {"neutral or dissatisfied": 0, "satisfied": 1}.get
    )

    return df_clean


# Aplicar o pipeline exatamente igual em ambos os conjuntos de dados
train_clean = preprocess_pipeline(train_data)
test_clean = preprocess_pipeline(test_data)

# =========================================================
# 7. SEPARAÇÃO DAS VARIÁVEIS (X e y)
# =========================================================
# Conjunto de Treino
X_train = train_clean.drop(columns=["satisfaction"])
y_train = train_clean["satisfaction"]

# Conjunto de Teste
X_test = test_clean.drop(columns=["satisfaction"])
y_test = test_clean["satisfaction"]

# Garantia absoluta de alinhamento de colunas (caso alguma categoria falte no teste)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nDimensão das matrizes finais após processamento:")
print(f"Treino -> X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Teste  -> X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# =========================================================
# 8. SCALING EXCLUSIVO PARA O SVM
# =========================================================
scaler = StandardScaler()

# O scaler aprende os parâmetros apenas com o X_train (evita data leakage)
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# =========================================================
# 9. TREINO DOS MODELOS SUPERVISIONADOS
# =========================================================
# --- SVM ---
# 1. Criar o modelo base sem o parâmetro probability
svm_base = SVC(kernel="rbf", random_state=42)

# 2. Envolver o modelo base no CalibratedClassifierCV (como sugerido pelo Scikit-Learn)
svm = CalibratedClassifierCV(estimator=svm_base, ensemble=False)  # type: ignore

print("\nA treinar o SVM (pode demorar alguns minutos)...")
svm.fit(X_train_svm, y_train)

# --- RANDOM FOREST ---
rf = RandomForestClassifier(n_estimators=300, random_state=42)
print("A treinar o Random Forest...")
rf.fit(X_train, y_train)

# --- XGBOOST ---
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss",
)
print("A treinar o XGBoost...")
xgb.fit(X_train, y_train)

# =========================================================
# 10. AVALIAÇÃO DOS MODELOS
# =========================================================
resultados = []


def avaliar_modelo(nome, modelo, X_teste):
    y_pred = modelo.predict(X_teste)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, modelo.predict_proba(X_teste)[:, 1])

    resultados.append([nome, accuracy, precision, recall, f1, roc_auc])

    print("\n====================================")
    print(nome)
    print("====================================")
    print(classification_report(y_test, y_pred))


# Executar a avaliação
avaliar_modelo("SVM", svm, X_test_svm)
avaliar_modelo("Random Forest", rf, X_test)
avaliar_modelo("XGBoost", xgb, X_test)

# =========================================================
# 11. COMPARAÇÃO DOS MODELOS
# =========================================================
resultado_df = pd.DataFrame(
    resultados, columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
)

print("\nComparação dos Modelos:")
print(resultado_df)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=resultado_df, x="Modelo", y="F1", hue="Modelo", palette="viridis", legend=False
)
plt.title("Comparação dos Modelos (F1-Score)")
plt.savefig("Graphs/F1-Score.png")
plt.close()
# =========================================================
# 12. ESCOLHA DO MELHOR MODELO E MATRIZ DE CONFUSÃO
# =========================================================
melhor_modelo_nome = resultado_df.loc[resultado_df["F1"].idxmax(), "Modelo"]
print(f"\nO Melhor Modelo baseado no F1-Score é: {melhor_modelo_nome}")

if melhor_modelo_nome == "SVM":
    melhor_modelo = svm
    X_final = X_test_svm
elif melhor_modelo_nome == "Random Forest":
    melhor_modelo = rf
    X_final = X_test
else:
    melhor_modelo = xgb
    X_final = X_test

y_pred_best = melhor_modelo.predict(X_final)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusão - {melhor_modelo_nome}")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig("Graphs/matriz-confusao.png")
plt.close()

# =========================================================
# 13. IMPORTÂNCIA DAS FEATURES
# =========================================================
if melhor_modelo_nome != "SVM":
    importancia = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importancia": melhor_modelo.feature_importances_,  # type: ignore
        }
    )

    importancia = importancia.sort_values(by="Importancia", ascending=False)

    print(f"\nTop 20 Features Mais Importantes ({melhor_modelo_nome}):")
    print(importancia.head(20))

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=importancia.head(20),
        x="Importancia",
        y="Feature",
        hue="Feature",
        palette="magma",
        legend=False,
    )
    plt.title(f"Importância das Features ({melhor_modelo_nome})")
    plt.savefig("Graphs/importancia-features-melhor-modelo.png")
    plt.close()
else:
    print(
        "\nO SVM foi o melhor modelo. Este algoritmo não disponibiliza feature importance diretamente."
    )


# =========================================================
# =========================================================
# 14. ML NÃO SUPERVISIONADO: STANDARDIZATION & PCA
# =========================================================
# =========================================================
print("\nPipeline Não Supervisionado (PCA + K-Means)...")

# Para PCA, precisamos de escalar todas as variáveis (usamos o X_train limpo)
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X_train)

# Aplicar PCA com 2 componentes para visualização 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

feature_names = X_train.columns
# PC = Principal Component
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=feature_names)  # type: ignore

print("\nVariância explicada pelos 2 componentes principais:")
print(pca.explained_variance_ratio_)

# =========================================================
# 15. VISUALIZAÇÃO DOS LOADINGS (HEATMAP & SCATTER)
# =========================================================
plt.figure(figsize=(10, 12))
sns.heatmap(loadings, cmap="coolwarm", center=0)
plt.title("Heatmap dos Loadings da PCA")
plt.savefig("Graphs/heatmap-loadings-pca.png")
plt.close()

plt.figure(figsize=(10, 8))
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

for i in loadings.index:
    plt.scatter(loadings.loc[i, "PC1"], loadings.loc[i, "PC2"])
    plt.text(
        loadings.loc[i, "PC1"] + 0.01, loadings.loc[i, "PC2"] + 0.01, i, fontsize=9
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Loadings das Variáveis (PC1 vs PC2)")
plt.savefig("Graphs/loading-variaveis-pc1-vs-pc2.png")
plt.close()

# =========================================================
# 16. K-MEANS NO ESPAÇO PCA
# =========================================================
# Vamos testar 4 clusters como base
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # type: ignore
clusters = kmeans.fit_predict(X_pca)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["cluster"] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="cluster", palette="Set2")
plt.title("Clusters dos Passageiros (PCA + KMeans)")
plt.savefig("Graphs/clusters-passageiros-pca-kmeans.png")
plt.close()

# =========================================================
# 17. PERFIL DOS CLUSTERS (VARIÁVEIS NUMÉRICAS)
# =========================================================
# Juntar os clusters ao dataset limpo (antes do one-hot encoding para ser legível)
train_cluster_profile = train_clean.copy()
train_cluster_profile["cluster"] = clusters

# Perfil médio numérico
cluster_profile_num = pd.DataFrame(
    train_cluster_profile.groupby("cluster").mean(numeric_only=True)
)
print("\nPerfil Médio Numérico dos Clusters:")
print(cluster_profile_num.T)

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_profile_num, cmap="coolwarm")
plt.title("Perfil Médio dos Clusters (Variáveis Numéricas)")
plt.savefig("Graphs/perfil-medio-clusters.png")
plt.close()

# =========================================================
# 18. PERFIL DOS CLUSTERS (VARIÁVEIS CATEGÓRICAS)
# =========================================================
# Usamos o train_data original para ver as labels de texto originais (ex: 'Business')
train_original_cat = train_data.copy()
train_original_cat["cluster"] = clusters

# Passamos uma lista com "object" e "str" para garantir compatibilidade futura, podia ser so str agora
categorical_cols = train_original_cat.select_dtypes(include=["object", "str"]).columns


def cluster_mode_profile(data, cluster_col, categorical_cols):
    profiles = {}
    for c in sorted(data[cluster_col].unique()):
        subset = data[data[cluster_col] == c]
        profile = {}
        for col in categorical_cols:
            # classe mais frequente
            mode_value = subset[col].mode()[0]
            # percentagem dessa classe
            freq = subset[col].value_counts(normalize=True).iloc[0]
            profile[col] = f"{mode_value} ({freq:.1%})"
        profiles[c] = profile
    return pd.DataFrame(profiles).T


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

cluster_cat_profile = cluster_mode_profile(
    train_original_cat, "cluster", categorical_cols
)

print("\n==============================")
print("PERFIL CATEGÓRICO DOS CLUSTERS")
print("==============================")
print(cluster_cat_profile.T)
