import pandas as pd
import numpy as np

from scipy.stats import poisson
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ARQUIVO = "dataset/BRA_processado.csv"

FEATURES = [
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


# ============================================================
# CARREGAR DADOS
# ============================================================

df = pd.read_csv(ARQUIVO)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=FEATURES + ["Res"]
)

df = df.sort_values(
    "Date"
).reset_index(drop=True)

print(
    f"Partidas processadas: {len(df)}"
)

print(
    f"Características base: {len(FEATURES)}"
)


# ============================================================
# DIVISÃO CRONOLÓGICA
# ============================================================

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

print(
    f"Treino: {len(df_train)}"
)

print(
    f"Validação: {len(df_val)}"
)

print(
    f"Teste: {len(df_test)}"
)


# ============================================================
# xG
# ============================================================

def calcular_xg(linha):

    # --------------------------------------------------------
    # TIME DA CASA
    # --------------------------------------------------------

    home_geral = (
        linha["Home_Gols_Media"]
        +
        linha["Away_Gols_Sofridos_Media"]
    ) / 2

    home_casa = (
        linha["Home_Gols_Casa_Media"]
        +
        linha["Away_Gols_Fora_Media"]
    ) / 2

    expected_home = (
        0.60 * home_geral
        +
        0.40 * home_casa
    )

    # --------------------------------------------------------
    # TIME VISITANTE
    # --------------------------------------------------------

    away_geral = (
        linha["Away_Gols_Media"]
        +
        linha["Home_Gols_Sofridos_Media"]
    ) / 2

    away_fora = (
        linha["Away_Gols_Fora_Media"]
        +
        linha["Home_Gols_Sofridos_Casa_Media"]
    ) / 2

    expected_away = (
        0.60 * away_geral
        +
        0.40 * away_fora
    )

    # Segurança
    expected_home = np.clip(
        expected_home,
        0.15,
        4.50
    )

    expected_away = np.clip(
        expected_away,
        0.15,
        4.50
    )

    return expected_home, expected_away


# ============================================================
# POISSON
# ============================================================

def calcular_poisson(
    expected_home,
    expected_away
):

    prob_home = 0.0
    prob_draw = 0.0
    prob_away = 0.0

    for gols_home in range(11):

        for gols_away in range(11):

            prob = (
                poisson.pmf(
                    gols_home,
                    expected_home
                )
                *
                poisson.pmf(
                    gols_away,
                    expected_away
                )
            )

            if gols_home > gols_away:

                prob_home += prob

            elif gols_home == gols_away:

                prob_draw += prob

            else:

                prob_away += prob

    total = (
        prob_home
        +
        prob_draw
        +
        prob_away
    )

    return (
        prob_home / total,
        prob_draw / total,
        prob_away / total
    )


# ============================================================
# PROBABILIDADES BASE
# ============================================================

def calcular_probabilidades_xg(
    expected_home,
    expected_away
):
    """
    A diferença de xG controla a disputa.

    xG praticamente iguais:
        empate recebe maior força.

    Diferença pequena:
        empate continua forte,
        mas o favorito começa a ganhar espaço.

    Diferença média:
        favorito passa a dominar.

    Diferença grande:
        favorito recebe a maior parte da probabilidade.

    Não existem faixas ou decisões do tipo
    "se diferença < 0.10".
    """

    diferenca = (
        expected_home
        -
        expected_away
    )

    distancia = abs(
        diferenca
    )

    # ========================================================
    # EMPATE
    # ========================================================
    #
    # Este é o ponto principal da nova versão.
    #
    # Com diferença zero:
    # aproximadamente 40%.
    #
    # Depois cai de maneira contínua.
    # ========================================================

    empate = (
        0.20
        +
        0.20 *
        np.exp(
            -1.40 *
            distancia
        )
    )

    # ========================================================
    # RESTANTE PARA CASA/FORA
    # ========================================================

    restante = (
        1.0
        -
        empate
    )

    # ========================================================
    # FAVORITISMO PELO xG
    # ========================================================

    intensidade = 2.4

    vantagem = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -intensidade *
                diferenca
            )
        )
    )

    prob_home = (
        restante
        *
        vantagem
    )

    prob_away = (
        restante
        *
        (
            1.0
            -
            vantagem
        )
    )

    # ========================================================
    # MANDO DE CAMPO
    # ========================================================

    bonus_casa = 0.025

    prob_home += bonus_casa
    prob_away -= bonus_casa

    prob_home = max(
        prob_home,
        0.005
    )

    prob_away = max(
        prob_away,
        0.005
    )

    # ========================================================
    # NORMALIZAÇÃO
    # ========================================================

    total = (
        prob_home
        +
        empate
        +
        prob_away
    )

    prob_home /= total
    empate /= total
    prob_away /= total

    return (
        prob_home,
        empate,
        prob_away
    )


# ============================================================
# FORÇA HISTÓRICA
# ============================================================

def calcular_historico(linha):

    home_strength = (
        0.45 *
        linha["Home_Aproveitamento_Casa"]

        +

        0.30 *
        linha["Home_Aproveitamento"]

        +

        0.15 *
        linha["Home_Gols_Media"]

        -

        0.10 *
        linha["Home_Gols_Sofridos_Media"]
    )

    away_strength = (
        0.45 *
        linha["Away_Aproveitamento_Fora"]

        +

        0.30 *
        linha["Away_Aproveitamento"]

        +

        0.15 *
        linha["Away_Gols_Media"]

        -

        0.10 *
        linha["Away_Gols_Sofridos_Media"]
    )

    diferenca = (
        home_strength
        -
        away_strength
    )

    diferenca = np.clip(
        diferenca,
        -2.0,
        2.0
    )

    vantagem = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -3.0 *
                diferenca
            )
        )
    )

    # Histórico de equilíbrio
    empate = (
        0.20
        +
        0.10 *
        np.exp(
            -1.5 *
            abs(diferenca)
        )
    )

    restante = (
        1.0
        -
        empate
    )

    home = (
        restante
        *
        vantagem
    )

    away = (
        restante
        *
        (
            1.0
            -
            vantagem
        )
    )

    total = (
        home
        +
        empate
        +
        away
    )

    return (
        home / total,
        empate / total,
        away / total,
        abs(diferenca)
    )


# ============================================================
# PREVISÃO
# ============================================================

def prever_partida(linha):

    # --------------------------------------------------------
    # xG
    # --------------------------------------------------------

    xg_home, xg_away = calcular_xg(
        linha
    )

    # --------------------------------------------------------
    # MODELO BASE
    # --------------------------------------------------------

    (
        xg_home_prob,
        xg_draw_prob,
        xg_away_prob
    ) = calcular_probabilidades_xg(
        xg_home,
        xg_away
    )

    # --------------------------------------------------------
    # POISSON
    # --------------------------------------------------------

    (
        poisson_home,
        poisson_draw,
        poisson_away
    ) = calcular_poisson(
        xg_home,
        xg_away
    )

    # --------------------------------------------------------
    # COMBINAÇÃO
    # --------------------------------------------------------

    peso_xg = 0.55
    peso_poisson = 0.45

    base_home = (
        peso_xg *
        xg_home_prob

        +

        peso_poisson *
        poisson_home
    )

    base_draw = (
        peso_xg *
        xg_draw_prob

        +

        peso_poisson *
        poisson_draw
    )

    base_away = (
        peso_xg *
        xg_away_prob

        +

        peso_poisson *
        poisson_away
    )

    # --------------------------------------------------------
    # HISTÓRICO
    # --------------------------------------------------------

    (
        hist_home,
        hist_draw,
        hist_away,
        diferenca_historica
    ) = calcular_historico(
        linha
    )

    # --------------------------------------------------------
    # PESO HISTÓRICO
    # --------------------------------------------------------

    peso_historico = (
        0.10
        +
        0.20 *
        (
            1.0
            -
            np.exp(
                -2.0 *
                diferenca_historica
            )
        )
    )

    peso_historico = np.clip(
        peso_historico,
        0.10,
        0.30
    )

    peso_base = (
        1.0
        -
        peso_historico
    )

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    prob_home = (
        peso_base *
        base_home

        +

        peso_historico *
        hist_home
    )

    prob_draw = (
        peso_base *
        base_draw

        +

        peso_historico *
        hist_draw
    )

    prob_away = (
        peso_base *
        base_away

        +

        peso_historico *
        hist_away
    )

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    total = (
        prob_home
        +
        prob_draw
        +
        prob_away
    )

    prob_home /= total
    prob_draw /= total
    prob_away /= total

    return (
        prob_home,
        prob_draw,
        prob_away,
        xg_home,
        xg_away,
        diferenca_historica,
        poisson_home,
        poisson_draw,
        poisson_away
    )


# ============================================================
# GERAR PREVISÕES
# ============================================================

def gerar_previsoes(dados):

    previsoes = []

    for _, linha in dados.iterrows():

        (
            prob_home,
            prob_draw,
            prob_away,
            _,
            _,
            _,
            _,
            _,
            _
        ) = prever_partida(
            linha
        )

        probabilidades = {
            "H": prob_home,
            "D": prob_draw,
            "A": prob_away
        }

        previsao = max(
            probabilidades,
            key=probabilidades.get
        )

        previsoes.append(
            previsao
        )

    return np.array(
        previsoes
    )


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar(
    nome,
    dados
):

    real = dados["Res"].values

    previsto = gerar_previsoes(
        dados
    )

    acc = accuracy_score(
        real,
        previsto
    )

    precision = precision_score(
        real,
        previsto,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        real,
        previsto,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        real,
        previsto,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0
    )

    f1_draw = f1_score(
        real,
        previsto,
        labels=["D"],
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 60)
    print(nome)
    print("=" * 60)

    print(
        f"Accuracy: {acc * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall: {recall * 100:.2f}%"
    )

    print(
        f"Macro F1: {f1 * 100:.2f}%"
    )

    print(
        f"F1 Empate: {f1_draw * 100:.2f}%"
    )

    print()
    print("Matriz de confusão:")

    print(
        confusion_matrix(
            real,
            previsto,
            labels=["H", "D", "A"]
        )
    )

    print()

    print(
        classification_report(
            real,
            previsto,
            labels=["H", "D", "A"],
            target_names=[
                "Casa",
                "Empate",
                "Fora"
            ],
            zero_division=0
        )
    )

    print(
        "Distribuição das previsões:"
    )

    contagem = pd.Series(
        previsto
    ).value_counts()

    for resultado in [
        "H",
        "D",
        "A"
    ]:

        quantidade = contagem.get(
            resultado,
            0
        )

        percentual = (
            quantidade
            /
            len(previsto)
            *
            100
        )

        print(
            f"{resultado}: "
            f"{quantidade} "
            f"({percentual:.2f}%)"
        )

    return previsto


# ============================================================
# VALIDAÇÃO
# ============================================================

prev_val = avaliar(
    "VALIDAÇÃO",
    df_val
)


# ============================================================
# TESTE
# ============================================================

prev_test = avaliar(
    "TESTE",
    df_test
)


# ============================================================
# xG MÉDIO DO TESTE
# ============================================================

xg_home_list = []
xg_away_list = []

for _, linha in df_test.iterrows():

    xg_h, xg_a = calcular_xg(
        linha
    )

    xg_home_list.append(
        xg_h
    )

    xg_away_list.append(
        xg_a
    )

print()
print("=" * 60)
print("xG MÉDIO NO TESTE")
print("=" * 60)

print(
    f"Casa: "
    f"{np.mean(xg_home_list):.2f}"
)

print(
    f"Fora: "
    f"{np.mean(xg_away_list):.2f}"
)


# ============================================================
# EXEMPLOS
# ============================================================

print()
print("=" * 60)
print("EXEMPLOS DE PREVISÃO")
print("=" * 60)

for _, linha in df_test.tail(10).iterrows():

    (
        prob_home,
        prob_draw,
        prob_away,
        xg_home,
        xg_away,
        dif_hist,
        poisson_home,
        poisson_draw,
        poisson_away
    ) = prever_partida(
        linha
    )

    probabilidades = {
        "H": prob_home,
        "D": prob_draw,
        "A": prob_away
    }

    resultado = max(
        probabilidades,
        key=probabilidades.get
    )

    print()
    print(
        f"{linha['Home']} x "
        f"{linha['Away']}"
    )

    print(
        f"Previsão: {resultado}"
    )

    print(
        f"Final: "
        f"Casa {prob_home * 100:.2f}% | "
        f"Empate {prob_draw * 100:.2f}% | "
        f"Fora {prob_away * 100:.2f}%"
    )

    print(
        f"xG: "
        f"{xg_home:.2f} x "
        f"{xg_away:.2f} "
        f"(diferença "
        f"{xg_home - xg_away:+.2f})"
    )

    print(
        f"Poisson: "
        f"{poisson_home * 100:.2f}% / "
        f"{poisson_draw * 100:.2f}% / "
        f"{poisson_away * 100:.2f}%"
    )

    print(
        f"Diferença histórica: "
        f"{dif_hist:.3f}"
    )


# ============================================================
# ANÁLISE DOS EMPATES POR xG
# ============================================================

print()
print("=" * 60)
print("EMPATES POR DIFERENÇA DE xG")
print("=" * 60)

faixas = [
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.30),
    (0.30, 0.50),
    (0.50, 1.00),
    (1.00, 99.0)
]

for minimo, maximo in faixas:

    jogos = 0
    empates_reais = 0
    empates_previstos = 0

    for _, linha in df_test.iterrows():

        xg_h, xg_a = calcular_xg(
            linha
        )

        diferenca = abs(
            xg_h - xg_a
        )

        if (
            minimo <= diferenca
            <
            maximo
        ):

            jogos += 1

            if linha["Res"] == "D":
                empates_reais += 1

            (
                prob_h,
                prob_d,
                prob_a,
                _,
                _,
                _,
                _,
                _,
                _
            ) = prever_partida(
                linha
            )

            previsao = max(
                {
                    "H": prob_h,
                    "D": prob_d,
                    "A": prob_a
                },
                key={
                    "H": prob_h,
                    "D": prob_d,
                    "A": prob_a
                }.get
            )

            if previsao == "D":
                empates_previstos += 1

    if jogos == 0:
        continue

    percentual_real = (
        empates_reais
        /
        jogos
        *
        100
    )

    percentual_previsto = (
        empates_previstos
        /
        jogos
        *
        100
    )

    print(
        f"{minimo:.2f}-{maximo:.2f}: "
        f"{jogos} jogos | "
        f"Empates reais: "
        f"{empates_reais} "
        f"({percentual_real:.2f}%) | "
        f"Empates previstos: "
        f"{empates_previstos} "
        f"({percentual_previsto:.2f}%)"
    )