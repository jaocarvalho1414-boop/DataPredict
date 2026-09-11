import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

arquivo = "dataset/BRA_processado.csv"

dados = pd.read_csv(arquivo)

print("Dataset carregado com sucesso!")
print(f"Partidas encontradas: {len(dados)}")
print()


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "Home_Gols_Media",
    "Home_Aproveitamento",
    "Away_Gols_Media",
    "Away_Aproveitamento",
    "Home_Aproveitamento_Casa",
    "Diferenca_Gols_Media",
    "Diferenca_Gols_Sofridos",
    "Diferenca_Aproveitamento",
    "Diferenca_Gols_Casa_Fora",
    "Diferenca_Aproveitamento_Casa_Fora",
    "Diferenca_Vitorias",
    "Home_Pontos_10",
    "Home_Aproveitamento_Casa_10",
    "Diferenca_Pontos_5",
    "Diferenca_Pontos_10",
    "Diferenca_Gols_Media_10",
    "Diferenca_Gols_Sofridos_Media_10",
    "Elo_Home",
    "Elo_Away",
    "Diferenca_Elo"
]


print(f"Características utilizadas: {len(features)}")
print()


# ============================================================
# 3. PREPARAR DADOS
# ============================================================

dados_modelo = dados.dropna(
    subset=features + ["Res"]
).copy()

X = dados_modelo[features]
y = dados_modelo["Res"]


# ============================================================
# 4. DIVISÃO CRONOLÓGICA
# ============================================================

ponto = int(len(dados_modelo) * 0.80)

X_train = X.iloc[:ponto]
X_test = X.iloc[ponto:]

y_train = y.iloc[:ponto]
y_test = y.iloc[ponto:]


print("=" * 70)
print("DIVISÃO DOS DADOS")
print("=" * 70)

print(f"Partidas para treinamento: {len(X_train)}")
print(f"Partidas para teste: {len(X_test)}")
print()


# ============================================================
# 5. CRIAR OS 3 MODELOS
# ============================================================

modelos = {

    "Logistic Regression": Pipeline([
        (
            "escala",
            StandardScaler()
        ),

        (
            "modelo",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),


    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),


    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}


# ============================================================
# 6. TREINAR E COMPARAR
# ============================================================

resultados = {}

previsoes_modelos = {}

print("=" * 70)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 70)


for nome, modelo in modelos.items():

    print()
    print("-" * 70)
    print(f"TESTANDO: {nome}")
    print("-" * 70)

    print("Treinando modelo...")

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    previsoes_modelos[nome] = previsoes


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        previsoes
    )

    precision = precision_score(
        y_test,
        previsoes,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        previsoes,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        previsoes,
        average="macro",
        zero_division=0
    )


    resultados[nome] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1
    }


    print()
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"Macro F1:  {f1:.2%}")


# ============================================================
# 7. RANKING FINAL
# ============================================================

print()
print("=" * 70)
print("RANKING FINAL DOS MODELOS")
print("=" * 70)
print()


tabela = pd.DataFrame(resultados).T

tabela = tabela.sort_values(
    by="Accuracy",
    ascending=False
)


print(
    tabela.to_string(
        formatters={
            "Accuracy": "{:.2%}".format,
            "Precision": "{:.2%}".format,
            "Recall": "{:.2%}".format,
            "Macro F1": "{:.2%}".format
        }
    )
)


# ============================================================
# 8. VENCEDOR
# ============================================================

vencedor = tabela.index[0]

print()
print("=" * 70)
print("🏆 VENCEDOR")
print("=" * 70)
print()

print(f"Modelo vencedor: {vencedor}")
print(f"Accuracy: {tabela.loc[vencedor, 'Accuracy']:.2%}")
print(f"Precision: {tabela.loc[vencedor, 'Precision']:.2%}")
print(f"Recall: {tabela.loc[vencedor, 'Recall']:.2%}")
print(f"Macro F1: {tabela.loc[vencedor, 'Macro F1']:.2%}")


# ============================================================
# 9. MATRIZ DE CONFUSÃO DO VENCEDOR
# ============================================================

previsoes_vencedor = previsoes_modelos[vencedor]


print()
print("=" * 70)
print("MATRIZ DE CONFUSÃO DO VENCEDOR")
print("=" * 70)
print()


matriz = confusion_matrix(
    y_test,
    previsoes_vencedor,
    labels=["H", "D", "A"]
)


matriz_df = pd.DataFrame(
    matriz,
    index=[
        "Real: Casa",
        "Real: Empate",
        "Real: Visitante"
    ],
    columns=[
        "Previsto: Casa",
        "Previsto: Empate",
        "Previsto: Visitante"
    ]
)


print(matriz_df)


# ============================================================
# 10. DESEMPENHO POR RESULTADO
# ============================================================

print()
print("=" * 70)
print("DESEMPENHO POR RESULTADO")
print("=" * 70)
print()


relatorio = classification_report(
    y_test,
    previsoes_vencedor,
    labels=["H", "D", "A"],
    target_names=[
        "Vitória da casa",
        "Empate",
        "Vitória visitante"
    ],
    zero_division=0
)


print(relatorio)


# ============================================================
# 11. DISTRIBUIÇÃO DAS PREVISÕES
# ============================================================

print("=" * 70)
print("DISTRIBUIÇÃO DAS PREVISÕES")
print("=" * 70)
print()


nomes_resultados = {
    "H": "Vitória da casa",
    "D": "Empate",
    "A": "Vitória visitante"
}


contagem = pd.Series(
    previsoes_vencedor
).value_counts()


total = len(previsoes_vencedor)


for resultado in ["H", "D", "A"]:

    quantidade = contagem.get(
        resultado,
        0
    )

    porcentagem = quantidade / total

    print(
        f"{nomes_resultados[resultado]}: "
        f"{quantidade} "
        f"({porcentagem:.2%})"
    )


# ============================================================
# 12. DISTRIBUIÇÃO REAL
# ============================================================

print()
print("=" * 70)
print("DISTRIBUIÇÃO REAL")
print("=" * 70)
print()


contagem_real = y_test.value_counts()

total_real = len(y_test)


for resultado in ["H", "D", "A"]:

    quantidade = contagem_real.get(
        resultado,
        0
    )

    porcentagem = quantidade / total_real

    print(
        f"{nomes_resultados[resultado]}: "
        f"{quantidade} "
        f"({porcentagem:.2%})"
    )


# ============================================================
# 13. FINAL
# ============================================================

print()
print("=" * 70)
print("TESTE CONCLUÍDO!")
print("=" * 70)
print()

print(
    f"O melhor modelo foi: {vencedor}"
)

print(
    f"Accuracy final: "
    f"{tabela.loc[vencedor, 'Accuracy']:.2%}"
)