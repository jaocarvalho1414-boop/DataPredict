
import pandas as pd
import numpy as np

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from math import factorial, exp


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_PROCESSADO = "dataset/BRA_processado.csv"
ARQUIVO_ORIGINAL = "dataset/BRA.csv"


# ============================================================
# 1. CARREGAMENTO
# ============================================================

print("Carregando dataset processado...")

df_proc = pd.read_csv(ARQUIVO_PROCESSADO)

print(f"Partidas encontradas: {len(df_proc)}")

print("\nCarregando resultados originais...")

df = pd.read_csv(ARQUIVO_ORIGINAL)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d/%m/%Y",
    errors="coerce"
)

df_proc["Date"] = pd.to_datetime(
    df_proc["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["HG", "AG", "Res"]
).copy()

print(f"Partidas após cruzamento: {len(df)}")


# ============================================================
# 2. CARACTERÍSTICAS BASE
# ============================================================

FEATURES_BASE = [
    "Home_Gols_Media",
    "Home_Gols_Sofridos_Media",
    "Home_Aproveitamento",

    "Away_Gols_Media",
    "Away_Gols_Sofridos_Media",
    "Away_Aproveitamento",

    "Home_Gols_Casa_Media",
    "Home_Gols_Sofridos_Casa_Media",
    "Home_Aproveitamento_Casa",

    "Away_Gols_Fora_Media",
    "Away_Gols_Sofridos_Fora_Media",
    "Away_Aproveitamento_Fora",

    "Home_Vitorias",
    "Home_Derrotas",

    "Away_Vitorias",
    "Away_Derrotas",

    "Diferenca_Gols_Media",
    "Diferenca_Gols_Sofridos",
    "Diferenca_Aproveitamento",

    "Diferenca_Gols_Casa_Fora"
]

FEATURES_BASE = [
    col for col in FEATURES_BASE
    if col in df_proc.columns
]

print(
    f"Características base utilizadas: "
    f"{len(FEATURES_BASE)}"
)


# ============================================================
# 3. PREPARAÇÃO DOS NOMES
# ============================================================

df["Home"] = df["Home"].astype(str)
df["Away"] = df["Away"].astype(str)

df_proc["Home"] = df_proc["Home"].astype(str)
df_proc["Away"] = df_proc["Away"].astype(str)


# ============================================================
# 4. CRUZAMENTO
# ============================================================

resultados = df[
    [
        "Date",
        "Home",
        "Away",
        "HG",
        "AG"
    ]
].copy()

dados = pd.merge(
    df_proc,
    resultados,
    on=[
        "Date",
        "Home",
        "Away"
    ],
    how="inner"
)

if "Res" not in dados.columns:

    dados["Res"] = np.where(
        dados["HG"] > dados["AG"],
        "H",
        np.where(
            dados["HG"] < dados["AG"],
            "A",
            "D"
        )
    )

print(
    f"Partidas após cruzamento: "
    f"{len(dados)}"
)


# ============================================================
# 5. ORDENAÇÃO
# ============================================================

dados = dados.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# 6. MATRIZES
# ============================================================

X_base = dados[
    FEATURES_BASE
].copy()

X_base = X_base.replace(
    [np.inf, -np.inf],
    np.nan
)

X_base = X_base.fillna(0)

y = dados["Res"].astype(str).values

gols_casa = dados["HG"].astype(float).values
gols_fora = dados["AG"].astype(float).values


# ============================================================
# 7. DIVISÃO CRONOLÓGICA
# ============================================================

n = len(dados)

limite_treino = int(n * 0.60)
limite_validacao = int(n * 0.80)

X_train = X_base.iloc[:limite_treino]
X_val = X_base.iloc[
    limite_treino:limite_validacao
]
X_test = X_base.iloc[
    limite_validacao:
]

y_train = y[:limite_treino]
y_val = y[
    limite_treino:limite_validacao
]
y_test = y[limite_validacao:]

gols_casa_train = gols_casa[:limite_treino]
gols_fora_train = gols_fora[:limite_treino]

gols_casa_val = gols_casa[
    limite_treino:limite_validacao
]
gols_fora_val = gols_fora[
    limite_treino:limite_validacao
]

gols_casa_test = gols_casa[
    limite_validacao:
]
gols_fora_test = gols_fora[
    limite_validacao:
]

print("\nDivisão dos dados:")

print(
    f"Treino:     {len(y_train)} partidas"
)

print(
    f"Validação:  {len(y_val)} partidas"
)

print(
    f"Teste:      {len(y_test)} partidas"
)


# ============================================================
# 8. POISSON
# ============================================================

def poisson_probabilidades(
    lambda_home,
    lambda_away,
    max_goals=8
):

    resultados = []

    for lh, la in zip(
        lambda_home,
        lambda_away
    ):

        probs_home = np.array([
            exp(-lh)
            * (lh ** g)
            / factorial(g)

            for g in range(
                max_goals + 1
            )
        ])

        probs_away = np.array([
            exp(-la)
            * (la ** g)
            / factorial(g)

            for g in range(
                max_goals + 1
            )
        ])

        matriz = np.outer(
            probs_home,
            probs_away
        )

        p_home = np.tril(
            matriz,
            -1
        ).sum()

        p_draw = np.trace(
            matriz
        )

        p_away = np.triu(
            matriz,
            1
        ).sum()

        total = (
            p_home
            + p_draw
            + p_away
        )

        resultados.append([
            p_home / total,
            p_draw / total,
            p_away / total
        ])

    return np.array(resultados)


# ============================================================
# 9. MODELOS DE GOLS
# ============================================================

print("\nTREINANDO MODELOS DE GOLS")

modelo_gols_casa = make_pipeline(
    StandardScaler(),
    PoissonRegressor(
        alpha=0.5,
        max_iter=1000
    )
)

modelo_gols_fora = make_pipeline(
    StandardScaler(),
    PoissonRegressor(
        alpha=0.5,
        max_iter=1000
    )
)

modelo_gols_casa.fit(
    X_train,
    gols_casa_train
)

modelo_gols_fora.fit(
    X_train,
    gols_fora_train
)


# ============================================================
# 10. GOLS ESPERADOS
# ============================================================

def calcular_xg(
    modelo_home,
    modelo_away,
    X
):

    xg_home = np.clip(
        modelo_home.predict(X),
        0.05,
        5
    )

    xg_away = np.clip(
        modelo_away.predict(X),
        0.05,
        5
    )

    return xg_home, xg_away


xg_home_train, xg_away_train = calcular_xg(
    modelo_gols_casa,
    modelo_gols_fora,
    X_train
)

xg_home_val, xg_away_val = calcular_xg(
    modelo_gols_casa,
    modelo_gols_fora,
    X_val
)

xg_home_test, xg_away_test = calcular_xg(
    modelo_gols_casa,
    modelo_gols_fora,
    X_test
)


# ============================================================
# 11. POISSON
# ============================================================

p_train = poisson_probabilidades(
    xg_home_train,
    xg_away_train
)

p_val = poisson_probabilidades(
    xg_home_val,
    xg_away_val
)

p_test = poisson_probabilidades(
    xg_home_test,
    xg_away_test
)


# ============================================================
# 12. LÓGICA PRINCIPAL DA DIFERENÇA
# ============================================================

def probabilidades_pela_diferenca(
    xg_home,
    xg_away
):

    diferenca = (
        xg_home - xg_away
    )

    distancia = np.abs(
        diferenca
    )

    # --------------------------------------------------------
    # EMPATE COMEÇA FORTE
    # --------------------------------------------------------
    #
    # Diferença = 0:
    # empate = 60%
    #
    # Conforme a diferença aumenta,
    # o empate perde espaço.
    #

    prob_draw = (
        0.60
        * np.exp(
            -1.20 * distancia
        )
    )

    # O empate nunca é eliminado
    # completamente.
    prob_draw = np.maximum(
        prob_draw,
        0.15
    )

    # --------------------------------------------------------
    # O QUE SOBROU VAI PARA CASA/FORA
    # --------------------------------------------------------

    restante = (
        1.0 - prob_draw
    )

    # Diferença positiva:
    # mais probabilidade para casa.
    #
    # Diferença negativa:
    # mais probabilidade para fora.

    vantagem_casa = (
        1
        /
        (
            1
            + np.exp(
                -5 * diferenca
            )
        )
    )

    prob_home = (
        restante
        * vantagem_casa
    )

    prob_away = (
        restante
        * (1 - vantagem_casa)
    )

    return np.column_stack([
        prob_home,
        prob_draw,
        prob_away
    ])


# ============================================================
# 13. DESEMPENHO HISTÓRICO
# ============================================================

def probabilidades_historicas(
    X
):

    home_geral = X[
        "Home_Aproveitamento"
    ].values

    away_geral = X[
        "Away_Aproveitamento"
    ].values

    home_casa = X[
        "Home_Aproveitamento_Casa"
    ].values

    away_fora = X[
        "Away_Aproveitamento_Fora"
    ].values

    # Mistura desempenho geral
    # com desempenho específico.

    forca_home = (
        home_geral * 0.60
        +
        home_casa * 0.40
    )

    forca_away = (
        away_geral * 0.60
        +
        away_fora * 0.40
    )

    diferenca = (
        forca_home
        - forca_away
    )

    # Distribuição da força
    vantagem_home = (
        1
        /
        (
            1
            + np.exp(
                -5 * diferenca
            )
        )
    )

    # Taxa histórica aproximada
    # de empates do campeonato.
    prob_draw = 0.265

    restante = (
        1 - prob_draw
    )

    prob_home = (
        restante
        * vantagem_home
    )

    prob_away = (
        restante
        * (1 - vantagem_home)
    )

    return np.column_stack([
        prob_home,
        np.full(len(X), prob_draw),
        prob_away
    ])


# ============================================================
# 14. COMBINAÇÃO xG + HISTÓRICO
# ============================================================

def probabilidades_finais(
    X,
    xg_home,
    xg_away
):

    probs_xg = probabilidades_pela_diferenca(
        xg_home,
        xg_away
    )

    probs_hist = probabilidades_historicas(
        X
    )

    # --------------------------------------------------------
    # DIFERENÇA HISTÓRICA
    # --------------------------------------------------------

    home_forca = (
        X["Home_Aproveitamento"].values
        * 0.60
        +
        X["Home_Aproveitamento_Casa"].values
        * 0.40
    )

    away_forca = (
        X["Away_Aproveitamento"].values
        * 0.60
        +
        X["Away_Aproveitamento_Fora"].values
        * 0.40
    )

    diferenca_historica = np.abs(
        home_forca - away_forca
    )

    # --------------------------------------------------------
    # PESO DO HISTÓRICO
    # --------------------------------------------------------
    #
    # Histórico equilibrado:
    # xG domina.
    #
    # Histórico muito diferente:
    # histórico ganha força.
    #

    peso_historico = np.clip(
        diferenca_historica / 0.30,
        0,
        1
    )

    peso_xg = (
        1 - peso_historico
    )

    probs = (
        probs_xg
        * peso_xg[:, None]
        +
        probs_hist
        * peso_historico[:, None]
    )

    # Normalização final
    probs = (
        probs
        /
        probs.sum(
            axis=1,
            keepdims=True
        )
    )

    return probs


# ============================================================
# 15. PROBABILIDADES DE VALIDAÇÃO
# ============================================================

probs_val = probabilidades_finais(
    X_val,
    xg_home_val,
    xg_away_val
)


# ============================================================
# 16. PREVISÃO
# ============================================================

def fazer_previsao(
    probabilidades
):

    classes = np.array([
        "H",
        "D",
        "A"
    ])

    return classes[
        np.argmax(
            probabilidades,
            axis=1
        )
    ]


pred_val = fazer_previsao(
    probs_val
)


# ============================================================
# 17. MÉTRICAS VALIDAÇÃO
# ============================================================

print(
    "\nRESULTADO - VALIDAÇÃO"
)

print(
    f"Accuracy:   "
    f"{accuracy_score(y_val, pred_val):.2%}"
)

print(
    f"Precision:  "
    f"{precision_score(y_val, pred_val, labels=['H','D','A'], average='macro', zero_division=0):.2%}"
)

print(
    f"Recall:     "
    f"{recall_score(y_val, pred_val, labels=['H','D','A'], average='macro', zero_division=0):.2%}"
)

print(
    f"Macro F1:   "
    f"{f1_score(y_val, pred_val, labels=['H','D','A'], average='macro', zero_division=0):.2%}"
)

print(
    f"F1 Empate:  "
    f"{f1_score(y_val, pred_val, labels=['D'], average='macro', zero_division=0):.2%}"
)


# ============================================================
# 18. DISTRIBUIÇÃO VALIDAÇÃO
# ============================================================

print(
    "\nDISTRIBUIÇÃO DAS PREVISÕES - VALIDAÇÃO"
)

print(
    f"Casa: "
    f"{np.sum(pred_val == 'H')} "
    f"({np.mean(pred_val == 'H'):.2%})"
)

print(
    f"Empate: "
    f"{np.sum(pred_val == 'D')} "
    f"({np.mean(pred_val == 'D'):.2%})"
)

print(
    f"Fora: "
    f"{np.sum(pred_val == 'A')} "
    f"({np.mean(pred_val == 'A'):.2%})"
)


# ============================================================
# 19. MATRIZ VALIDAÇÃO
# ============================================================

print(
    "\nMATRIZ DE CONFUSÃO - VALIDAÇÃO"
)

print(
    confusion_matrix(
        y_val,
        pred_val,
        labels=[
            "H",
            "D",
            "A"
        ]
    )
)


# ============================================================
# 20. RETREINAMENTO FINAL
# ============================================================

print(
    "\nRETREINANDO MODELOS DE GOLS "
    "COM TREINO + VALIDAÇÃO"
)

X_trainval = pd.concat([
    X_train,
    X_val
])

gols_casa_trainval = np.concatenate([
    gols_casa_train,
    gols_casa_val
])

gols_fora_trainval = np.concatenate([
    gols_fora_train,
    gols_fora_val
])


modelo_gols_casa_final = make_pipeline(
    StandardScaler(),
    PoissonRegressor(
        alpha=0.5,
        max_iter=1000
    )
)

modelo_gols_fora_final = make_pipeline(
    StandardScaler(),
    PoissonRegressor(
        alpha=0.5,
        max_iter=1000
    )
)

modelo_gols_casa_final.fit(
    X_trainval,
    gols_casa_trainval
)

modelo_gols_fora_final.fit(
    X_trainval,
    gols_fora_trainval
)


# ============================================================
# 21. xG FINAL
# ============================================================

xg_home_test_final, xg_away_test_final = (
    calcular_xg(
        modelo_gols_casa_final,
        modelo_gols_fora_final,
        X_test
    )
)


# ============================================================
# 22. POISSON FINAL
# ============================================================

p_test_final = poisson_probabilidades(
    xg_home_test_final,
    xg_away_test_final
)


# ============================================================
# 23. PROBABILIDADES FINAIS
# ============================================================

probs_test = probabilidades_finais(
    X_test,
    xg_home_test_final,
    xg_away_test_final
)

pred_test = fazer_previsao(
    probs_test
)


# ============================================================
# 24. RESULTADO FINAL
# ============================================================

print(
    "\nRESULTADO FINAL - TESTE"
)

print(
    f"Accuracy:   "
    f"{accuracy_score(y_test, pred_test):.2%}"
)

print(
    f"Precision:  "
    f"{precision_score(y_test, pred_test, labels=['H','D','A'], average='macro', zero_division=0):.2%}"
)

print(
    f"Recall:     "
    f"{recall_score(y_test, pred_test, labels=['H','D','A'], average='macro', zero_division=0):.2%}"
)

print(
    f"Macro F1:   "
    f"{f1_score(y_test, pred_test, labels=['H','D','A'], average='macro', zero_division=0):.2%}"
)

print(
    f"F1 Empate:  "
    f"{f1_score(y_test, pred_test, labels=['D'], average='macro', zero_division=0):.2%}"
)


# ============================================================
# 25. MATRIZ DE CONFUSÃO
# ============================================================

print(
    "\nMATRIZ DE CONFUSÃO"
)

print(
    confusion_matrix(
        y_test,
        pred_test,
        labels=[
            "H",
            "D",
            "A"
        ]
    )
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print(
    "\nCLASSIFICATION REPORT"
)

print(
    classification_report(
        y_test,
        pred_test,
        labels=[
            "H",
            "D",
            "A"
        ],
        target_names=[
            "Casa",
            "Empate",
            "Fora"
        ],
        zero_division=0
    )
)


# ============================================================
# 27. DISTRIBUIÇÃO TESTE
# ============================================================

print(
    "\nDISTRIBUIÇÃO TESTE"
)

print(
    f"Casa: "
    f"{np.sum(pred_test == 'H')} "
    f"({np.mean(pred_test == 'H'):.2%})"
)

print(
    f"Empate: "
    f"{np.sum(pred_test == 'D')} "
    f"({np.mean(pred_test == 'D'):.2%})"
)

print(
    f"Fora: "
    f"{np.sum(pred_test == 'A')} "
    f"({np.mean(pred_test == 'A'):.2%})"
)


# ============================================================
# 28. MODELO DE GOLS
# ============================================================

print(
    "\nMODELO DE GOLS"
)

print(
    f"Média prevista home: "
    f"{np.mean(xg_home_test_final):.2f}"
)

print(
    f"Média prevista away: "
    f"{np.mean(xg_away_test_final):.2f}"
)


# ============================================================
# 29. PLACAR MAIS PROVÁVEL
# ============================================================

def placar_mais_provavel(
    lambda_home,
    lambda_away,
    max_goals=8
):

    melhor_prob = -1
    melhor_placar = (0, 0)

    for h in range(
        max_goals + 1
    ):

        ph = (
            exp(-lambda_home)
            * (lambda_home ** h)
            / factorial(h)
        )

        for a in range(
            max_goals + 1
        ):

            pa = (
                exp(-lambda_away)
                * (lambda_away ** a)
                / factorial(a)
            )

            prob = ph * pa

            if prob > melhor_prob:

                melhor_prob = prob

                melhor_placar = (
                    h,
                    a
                )

    return melhor_placar


# ============================================================
# 30. EXEMPLOS
# ============================================================

print(
    "\nEXEMPLOS DE PREVISÃO"
)

indices_exemplos = np.linspace(
    0,
    len(X_test) - 1,
    min(10, len(X_test)),
    dtype=int
)

for idx in indices_exemplos:

    linha = dados.iloc[
        limite_validacao + idx
    ]

    probs = probs_test[idx]

    xg_h = xg_home_test_final[idx]
    xg_a = xg_away_test_final[idx]

    diferenca = (
        xg_h - xg_a
    )

    placar = placar_mais_provavel(
        xg_h,
        xg_a
    )

    print(
        "\n"
        + str(linha["Home"])
        + " x "
        + str(linha["Away"])
    )

    print(
        f"Casa: "
        f"{probs[0] * 100:.2f}%"
    )

    print(
        f"Empate: "
        f"{probs[1] * 100:.2f}%"
    )

    print(
        f"Fora: "
        f"{probs[2] * 100:.2f}%"
    )

    print(
        f"Gols esperados: "
        f"{xg_h:.2f} x "
        f"{xg_a:.2f}"
    )

    print(
        f"Diferença xG: "
        f"{diferenca:+.2f}"
    )

    print(
        f"Poisson: "
        f"H {p_test_final[idx, 0] * 100:.2f}% | "
        f"D {p_test_final[idx, 1] * 100:.2f}% | "
        f"A {p_test_final[idx, 2] * 100:.2f}%"
    )

    print(
        f"Placar mais provável: "
        f"{placar[0]}x{placar[1]}"
    )


# ============================================================
# 31. ANÁLISE DA DIFERENÇA
# ============================================================

print(
    "\nANÁLISE DA RELAÇÃO ENTRE "
    "DIFERENÇA DE xG E EMPATES"
)

diferencas = np.abs(
    xg_home_test_final
    - xg_away_test_final
)

faixas = [
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.30),
    (0.30, 0.50),
    (0.50, 1.00),
    (1.00, np.inf)
]

for inicio, fim in faixas:

    if np.isinf(fim):

        mask = (
            diferencas >= inicio
        )

    else:

        mask = (
            (diferencas >= inicio)
            &
            (diferencas < fim)
        )

    quantidade = np.sum(mask)

    if quantidade == 0:
        continue

    reais = np.sum(
        y_test[mask] == "D"
    )

    previstos = np.sum(
        pred_test[mask] == "D"
    )

    if np.isinf(fim):

        nome = f"{inicio:.2f}–+"

    else:

        nome = (
            f"{inicio:.2f}–{fim:.2f}"
        )

    print(
        f"{nome}: "
        f"{quantidade} jogos | "
        f"empates reais: {reais} "
        f"({reais / quantidade:.2%}) | "
        f"empates previstos: {previstos} "
        f"({previstos / quantidade:.2%})"
    )


print(
    "\nMODELO FINAL CONCLUÍDO."
)

