
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_PROCESSADO = "dataset/BRA_processado.csv"
ARQUIVO_ORIGINAL = "dataset/BRA.csv"

FEATURES = [
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

CLASSES = ["H", "D", "A"]


# ============================================================
# PROBABILIDADES DO HGB NA ORDEM H / D / A
# ============================================================

def probabilidades_hda(modelo, X):

    probabilidades = modelo.predict_proba(X)

    resultado = np.zeros((len(X), 3))

    for i, classe in enumerate(modelo.classes_):

        if classe == "H":
            resultado[:, 0] = probabilidades[:, i]

        elif classe == "D":
            resultado[:, 1] = probabilidades[:, i]

        elif classe == "A":
            resultado[:, 2] = probabilidades[:, i]

    return resultado


# ============================================================
# POISSON
# ============================================================

def probabilidade_poisson(gols, media):

    return (
        math.exp(-media)
        * (media ** gols)
        / math.factorial(gols)
    )


def probabilidades_poisson(
    media_casa,
    media_fora,
    max_gols=8
):

    prob_casa = np.array([
        probabilidade_poisson(
            g,
            media_casa
        )
        for g in range(max_gols + 1)
    ])

    prob_fora = np.array([
        probabilidade_poisson(
            g,
            media_fora
        )
        for g in range(max_gols + 1)
    ])

    prob_casa /= prob_casa.sum()
    prob_fora /= prob_fora.sum()

    prob_vitoria_casa = 0.0
    prob_empate = 0.0
    prob_vitoria_fora = 0.0

    for gols_casa in range(max_gols + 1):

        for gols_fora in range(max_gols + 1):

            prob_placar = (
                prob_casa[gols_casa]
                * prob_fora[gols_fora]
            )

            if gols_casa > gols_fora:

                prob_vitoria_casa += prob_placar

            elif gols_casa == gols_fora:

                prob_empate += prob_placar

            else:

                prob_vitoria_fora += prob_placar

    probabilidades = np.array([
        prob_vitoria_casa,
        prob_empate,
        prob_vitoria_fora
    ])

    return probabilidades / probabilidades.sum()


# ============================================================
# AJUSTE DE EQUILÍBRIO
# ============================================================

def ajustar_empate(
    probabilidades,
    media_casa,
    media_fora,
    forca,
    escala
):
    """
    Aumenta a probabilidade de empate quando os gols
    esperados estão próximos.

    forca:
        intensidade do ajuste.

    escala:
        controla a velocidade com que o efeito diminui
        conforme a diferença de gols aumenta.
    """

    resultado = probabilidades.copy()

    diferenca = abs(
        media_casa - media_fora
    )

    # Quanto menor a diferença, maior o fator.
    equilibrio = math.exp(
        -diferenca / escala
    )

    # O empate recebe um multiplicador adicional.
    multiplicador = 1.0 + (
        forca * equilibrio
    )

    resultado[1] *= multiplicador

    # Normalização
    resultado /= resultado.sum()

    return resultado


# ============================================================
# APLICA AJUSTE PARA TODAS AS PARTIDAS
# ============================================================

def ajustar_todas(
    probabilidades,
    medias_casa,
    medias_fora,
    forca,
    escala
):

    resultado = []

    for prob, casa, fora in zip(
        probabilidades,
        medias_casa,
        medias_fora
    ):

        nova_prob = ajustar_empate(
            prob,
            casa,
            fora,
            forca,
            escala
        )

        resultado.append(nova_prob)

    return np.array(resultado)


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar(
    y_real,
    probabilidades
):

    previsoes = np.array(CLASSES)[
        np.argmax(
            probabilidades,
            axis=1
        )
    ]

    accuracy = accuracy_score(
        y_real,
        previsoes
    )

    precision = precision_score(
        y_real,
        previsoes,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_real,
        previsoes,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_real,
        previsoes,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        previsoes
    )


# ============================================================
# CARREGAMENTO
# ============================================================

print("Carregando dataset processado...")

df = pd.read_csv(
    ARQUIVO_PROCESSADO
)

print(
    f"Partidas encontradas: {len(df)}"
)

print("\nCarregando resultados originais...")

df_original = pd.read_csv(
    ARQUIVO_ORIGINAL
)

df_original["Date"] = pd.to_datetime(
    df_original["Date"],
    dayfirst=True,
    errors="coerce"
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# ============================================================
# RESULTADOS ORIGINAIS
# ============================================================

dados_resultados = df_original[
    [
        "Date",
        "Home",
        "Away",
        "HG",
        "AG"
    ]
].copy()

dados_resultados = dados_resultados.dropna(
    subset=[
        "Date",
        "Home",
        "Away",
        "HG",
        "AG"
    ]
)


# ============================================================
# CRUZAMENTO
# ============================================================

df = df.merge(
    dados_resultados,
    on=[
        "Date",
        "Home",
        "Away"
    ],
    how="inner"
)

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)

print(
    f"Partidas após cruzamento: {len(df)}"
)


# ============================================================
# FEATURES
# ============================================================

for feature in FEATURES:

    if feature not in df.columns:

        raise ValueError(
            f"A característica '{feature}' "
            "não foi encontrada."
        )

print(
    f"Características utilizadas: "
    f"{len(FEATURES)}"
)


# ============================================================
# DADOS
# ============================================================

X = df[FEATURES].copy()

y_resultado = df["Res"].astype(str)

y_gols_casa = df["HG"].astype(float)

y_gols_fora = df["AG"].astype(float)


# ============================================================
# DIVISÃO 60 / 20 / 20
# ============================================================

n = len(df)

tamanho_treino = int(
    n * 0.60
)

tamanho_validacao = int(
    n * 0.20
)

fim_treino = tamanho_treino

fim_validacao = (
    tamanho_treino
    +
    tamanho_validacao
)


X_treino = X.iloc[
    :fim_treino
]

X_validacao = X.iloc[
    fim_treino:fim_validacao
]

X_teste = X.iloc[
    fim_validacao:
]


y_treino = y_resultado.iloc[
    :fim_treino
]

y_validacao = y_resultado.iloc[
    fim_treino:fim_validacao
]

y_teste = y_resultado.iloc[
    fim_validacao:
]


gols_casa_treino = y_gols_casa.iloc[
    :fim_treino
]

gols_fora_treino = y_gols_fora.iloc[
    :fim_treino
]


print("\nDivisão dos dados:")
print(
    f"Treino:     {len(X_treino)} partidas"
)

print(
    f"Validação:  {len(X_validacao)} partidas"
)

print(
    f"Teste:      {len(X_teste)} partidas"
)


# ============================================================
# HGB
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TREINANDO HISTGRADIENTBOOSTING"
)

print(
    "=" * 60
)

modelo_hgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=7,
    min_samples_leaf=30,
    l2_regularization=2.0,
    random_state=42
)

modelo_hgb.fit(
    X_treino,
    y_treino
)

print(
    "Classes detectadas pelo HGB:"
)

print(
    modelo_hgb.classes_
)


# ============================================================
# MODELOS DE GOLS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TREINANDO MODELOS DE GOLS"
)

print(
    "=" * 60
)

modelo_gols_casa = PoissonRegressor(
    alpha=1.0,
    max_iter=3000,
    tol=1e-7
)

modelo_gols_fora = PoissonRegressor(
    alpha=1.0,
    max_iter=3000,
    tol=1e-7
)

modelo_gols_casa.fit(
    X_treino,
    gols_casa_treino
)

modelo_gols_fora.fit(
    X_treino,
    gols_fora_treino
)


# ============================================================
# GOLS ESPERADOS - VALIDAÇÃO
# ============================================================

media_casa_validacao = np.clip(
    modelo_gols_casa.predict(
        X_validacao
    ),
    0.01,
    8.0
)

media_fora_validacao = np.clip(
    modelo_gols_fora.predict(
        X_validacao
    ),
    0.01,
    8.0
)


# ============================================================
# POISSON - VALIDAÇÃO
# ============================================================

prob_poisson_validacao = np.array([

    probabilidades_poisson(
        casa,
        fora
    )

    for casa, fora

    in zip(
        media_casa_validacao,
        media_fora_validacao
    )
])


# ============================================================
# HGB - VALIDAÇÃO
# ============================================================

prob_hgb_validacao = probabilidades_hda(
    modelo_hgb,
    X_validacao
)


# ============================================================
# COMBINAÇÃO HGB + POISSON
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TESTANDO HGB + POISSON"
)

print(
    "=" * 60
)

melhor_peso_hgb = 0.0
melhor_score = -1

melhor_probabilidades_base = None


for peso_hgb in np.arange(
    0.0,
    1.01,
    0.05
):

    peso_poisson = (
        1.0 - peso_hgb
    )

    probabilidades = (
        peso_hgb
        * prob_hgb_validacao
        +
        peso_poisson
        * prob_poisson_validacao
    )

    acc, prec, rec, f1, _ = avaliar(
        y_validacao,
        probabilidades
    )

    score = (
        0.70 * acc
        +
        0.30 * f1
    )

    print(
        f"HGB={peso_hgb:.2f} | "
        f"Poisson={peso_poisson:.2f} | "
        f"Accuracy={acc * 100:.2f}% | "
        f"Macro F1={f1 * 100:.2f}%"
    )

    if score > melhor_score:

        melhor_score = score

        melhor_peso_hgb = (
            peso_hgb
        )

        melhor_probabilidades_base = (
            probabilidades.copy()
        )


melhor_peso_poisson = (
    1.0 - melhor_peso_hgb
)


# ============================================================
# MELHOR COMBINAÇÃO BASE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MELHOR COMBINAÇÃO BASE"
)

print(
    "=" * 60
)

print(
    f"Peso HGB:      "
    f"{melhor_peso_hgb:.2f}"
)

print(
    f"Peso Poisson:  "
    f"{melhor_peso_poisson:.2f}"
)


# ============================================================
# OTIMIZANDO AJUSTE DO EMPATE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "OTIMIZANDO AJUSTE DO EMPATE"
)

print(
    "=" * 60
)

melhor_forca = 0.0
melhor_escala = 1.0
melhor_score_empate = -1

melhor_resultado_ajustado = None


# Intensidade do aumento da probabilidade de empate
forca_testes = np.arange(
    0.0,
    2.01,
    0.10
)


# Velocidade do efeito da diferença
escala_testes = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00
]


for forca in forca_testes:

    for escala in escala_testes:

        probabilidades_ajustadas = (
            ajustar_todas(
                melhor_probabilidades_base,
                media_casa_validacao,
                media_fora_validacao,
                forca,
                escala
            )
        )

        acc, prec, rec, f1, previsoes = avaliar(
            y_validacao,
            probabilidades_ajustadas
        )

        score = (
            0.60 * acc
            +
            0.40 * f1
        )

        if score > melhor_score_empate:

            melhor_score_empate = score

            melhor_forca = forca

            melhor_escala = escala

            melhor_resultado_ajustado = (
                probabilidades_ajustadas.copy()
            )


# ============================================================
# RESULTADO DO AJUSTE
# ============================================================

acc, prec, rec, f1, previsoes = avaliar(
    y_validacao,
    melhor_resultado_ajustado
)


print(
    "\nMelhor ajuste encontrado:"
)

print(
    f"Força do empate: "
    f"{melhor_forca:.2f}"
)

print(
    f"Escala: "
    f"{melhor_escala:.2f}"
)

print(
    f"Accuracy: "
    f"{acc * 100:.2f}%"
)

print(
    f"Precision: "
    f"{prec * 100:.2f}%"
)

print(
    f"Recall: "
    f"{rec * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# DISTRIBUIÇÃO DA VALIDAÇÃO
# ============================================================

print(
    "\nDistribuição das previsões "
    "na validação:"
)

for classe, nome in zip(
    CLASSES,
    [
        "Vitória em casa",
        "Empate",
        "Vitória fora"
    ]
):

    quantidade = np.sum(
        previsoes == classe
    )

    percentual = (
        quantidade
        /
        len(previsoes)
        *
        100
    )

    print(
        f"{nome}: "
        f"{quantidade} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# RETREINO FINAL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "RETREINANDO MODELOS"
)

print(
    "=" * 60
)

X_treino_final = X.iloc[
    :fim_validacao
]

y_treino_final = y_resultado.iloc[
    :fim_validacao
]

gols_casa_treino_final = y_gols_casa.iloc[
    :fim_validacao
]

gols_fora_treino_final = y_gols_fora.iloc[
    :fim_validacao
]


# ============================================================
# HGB FINAL
# ============================================================

modelo_hgb_final = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=7,
    min_samples_leaf=30,
    l2_regularization=2.0,
    random_state=42
)

modelo_hgb_final.fit(
    X_treino_final,
    y_treino_final
)


# ============================================================
# POISSON FINAL
# ============================================================

modelo_gols_casa_final = PoissonRegressor(
    alpha=1.0,
    max_iter=3000,
    tol=1e-7
)

modelo_gols_fora_final = PoissonRegressor(
    alpha=1.0,
    max_iter=3000,
    tol=1e-7
)

modelo_gols_casa_final.fit(
    X_treino_final,
    gols_casa_treino_final
)

modelo_gols_fora_final.fit(
    X_treino_final,
    gols_fora_treino_final
)


# ============================================================
# HGB - TESTE
# ============================================================

prob_hgb_teste = probabilidades_hda(
    modelo_hgb_final,
    X_teste
)


# ============================================================
# GOLS - TESTE
# ============================================================

media_casa_teste = np.clip(
    modelo_gols_casa_final.predict(
        X_teste
    ),
    0.01,
    8.0
)

media_fora_teste = np.clip(
    modelo_gols_fora_final.predict(
        X_teste
    ),
    0.01,
    8.0
)


# ============================================================
# POISSON - TESTE
# ============================================================

prob_poisson_teste = np.array([

    probabilidades_poisson(
        casa,
        fora
    )

    for casa, fora

    in zip(
        media_casa_teste,
        media_fora_teste
    )
])


# ============================================================
# COMBINAÇÃO FINAL
# ============================================================

probabilidades_teste_base = (
    melhor_peso_hgb
    * prob_hgb_teste
    +
    melhor_peso_poisson
    * prob_poisson_teste
)


# ============================================================
# AJUSTE DE EMPATE NO TESTE
# ============================================================

probabilidades_teste = (
    ajustar_todas(
        probabilidades_teste_base,
        media_casa_teste,
        media_fora_teste,
        melhor_forca,
        melhor_escala
    )
)


# ============================================================
# RESULTADO FINAL
# ============================================================

acc, prec, rec, f1, previsoes_teste = avaliar(
    y_teste,
    probabilidades_teste
)


print(
    "\n" + "=" * 60
)

print(
    "RESULTADO FINAL - TESTE"
)

print(
    "=" * 60
)

print(
    f"Accuracy:   "
    f"{acc * 100:.2f}%"
)

print(
    f"Precision:  "
    f"{prec * 100:.2f}%"
)

print(
    f"Recall:     "
    f"{rec * 100:.2f}%"
)

print(
    f"Macro F1:   "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

matriz = confusion_matrix(
    y_teste,
    previsoes_teste,
    labels=CLASSES
)


print(
    "\n" + "=" * 60
)

print(
    "MATRIZ DE CONFUSÃO"
)

print(
    "=" * 60
)

print(
    "              Previsto: Casa  "
    "Previsto: Empate  "
    "Previsto: Fora"
)

print(
    f"Real: Casa     "
    f"{matriz[0,0]:>13}        "
    f"{matriz[0,1]:>13}        "
    f"{matriz[0,2]:>13}"
)

print(
    f"Real: Empate   "
    f"{matriz[1,0]:>13}        "
    f"{matriz[1,1]:>13}        "
    f"{matriz[1,2]:>13}"
)

print(
    f"Real: Fora     "
    f"{matriz[2,0]:>13}        "
    f"{matriz[2,1]:>13}        "
    f"{matriz[2,2]:>13}"
)


# ============================================================
# DISTRIBUIÇÃO
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "DISTRIBUIÇÃO DAS PREVISÕES"
)

print(
    "=" * 60
)

for classe, nome in zip(
    CLASSES,
    [
        "Vitória em casa",
        "Empate",
        "Vitória fora"
    ]
):

    quantidade = np.sum(
        previsoes_teste == classe
    )

    percentual = (
        quantidade
        /
        len(previsoes_teste)
        *
        100
    )

    print(
        f"{nome}: "
        f"{quantidade} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 60
)

print(
    classification_report(
        y_teste,
        previsoes_teste,
        labels=CLASSES,
        target_names=[
            "Vitória em casa",
            "Empate",
            "Vitória fora"
        ],
        zero_division=0
    )
)


# ============================================================
# MODELO DE GOLS
# ============================================================

print(
    "=" * 60
)

print(
    "MODELO DE GOLS"
)

print(
    "=" * 60
)

print(
    f"Média prevista de gols em casa: "
    f"{media_casa_teste.mean():.2f}"
)

print(
    f"Média prevista de gols fora: "
    f"{media_fora_teste.mean():.2f}"
)


# ============================================================
# EXEMPLOS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EXEMPLOS DE PROBABILIDADES"
)

print(
    "=" * 60
)

for i in range(
    min(5, len(X_teste))
):

    prob = probabilidades_teste[i]

    diferenca = abs(
        media_casa_teste[i]
        -
        media_fora_teste[i]
    )

    print(
        f"\nPartida: "
        f"{df.iloc[fim_validacao + i]['Home']} "
        f"x "
        f"{df.iloc[fim_validacao + i]['Away']}"
    )

    print(
        f"Vitória em casa: "
        f"{prob[0] * 100:.2f}%"
    )

    print(
        f"Empate:          "
        f"{prob[1] * 100:.2f}%"
    )

    print(
        f"Vitória fora:    "
        f"{prob[2] * 100:.2f}%"
    )

    print(
        f"Gols esperados: "
        f"{media_casa_teste[i]:.2f} "
        f"x "
        f"{media_fora_teste[i]:.2f}"
    )

    print(
        f"Diferença esperada: "
        f"{diferenca:.2f}"
    )


print(
    "\nProcessamento concluído!"
)

