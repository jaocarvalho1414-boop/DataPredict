import pandas as pd
from collections import defaultdict, deque

# =========================
# CONFIGURAÇÕES
# =========================

ARQUIVO_ENTRADA = "dataset/BRA.csv"
ARQUIVO_SAIDA = "dataset/BRA_processado.csv"

ELO_INICIAL = 1500
ELO_K = 20
VANTAGEM_CASA = 60


# =========================
# FUNÇÕES AUXILIARES
# =========================

def media(valores):
    if not valores:
        return 0
    return sum(valores) / len(valores)


def aproveitamento(resultados):
    if not resultados:
        return 0

    pontos = sum(resultados)
    return pontos / (len(resultados) * 3)


def ultimos_pontos(resultados, quantidade):
    if not resultados:
        return 0

    ultimos = resultados[-quantidade:]
    return sum(ultimos)


def gols_media(historico, quantidade=None):
    if not historico:
        return 0

    dados = historico if quantidade is None else historico[-quantidade:]
    return media([x[0] for x in dados])


def gols_sofridos_media(historico, quantidade=None):
    if not historico:
        return 0

    dados = historico if quantidade is None else historico[-quantidade:]
    return media([x[1] for x in dados])


def aproveitamento_historico(historico, quantidade=None):
    if not historico:
        return 0

    dados = historico if quantidade is None else historico[-quantidade:]
    resultados = [x[2] for x in dados]

    return aproveitamento(resultados)


def media_gols_casa(historico, quantidade=None):
    if not historico:
        return 0

    dados = historico if quantidade is None else historico[-quantidade:]
    return media([x[0] for x in dados])


def media_gols_sofridos_casa(historico, quantidade=None):
    if not historico:
        return 0

    dados = historico if quantidade is None else historico[-quantidade:]
    return media([x[1] for x in dados])


def aproveitamento_casa(historico, quantidade=None):
    if not historico:
        return 0

    dados = historico if quantidade is None else historico[-quantidade:]
    resultados = [x[2] for x in dados]

    return aproveitamento(resultados)


# =========================
# ELO
# =========================

def expectativa_elo(elo_casa, elo_fora):
    return 1 / (1 + 10 ** (-(elo_casa - elo_fora) / 400))


def atualizar_elo(elo_casa, elo_fora, resultado):
    expectativa_casa = expectativa_elo(
        elo_casa + VANTAGEM_CASA,
        elo_fora
    )

    if resultado == "H":
        resultado_real = 1
    elif resultado == "D":
        resultado_real = 0.5
    else:
        resultado_real = 0

    novo_elo_casa = elo_casa + ELO_K * (
        resultado_real - expectativa_casa
    )

    novo_elo_fora = elo_fora + ELO_K * (
        (1 - resultado_real) - (1 - expectativa_casa)
    )

    return novo_elo_casa, novo_elo_fora


# =========================
# CARREGAMENTO
# =========================

print("Carregando dataset...")

df = pd.read_csv(ARQUIVO_ENTRADA)

print(f"Partidas encontradas: {len(df)}")

# Remover partidas sem resultado
df = df.dropna(subset=["HG", "AG", "Res"]).copy()

# Converter data
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

# Ordenar cronologicamente
df = df.sort_values("Date").reset_index(drop=True)


# =========================
# HISTÓRICO DOS TIMES
# =========================

historico = defaultdict(list)

historico_casa = defaultdict(list)

historico_fora = defaultdict(list)

elo = defaultdict(lambda: ELO_INICIAL)


# =========================
# PROCESSAMENTO
# =========================

dados_processados = []

for _, jogo in df.iterrows():

    casa = jogo["Home"]
    fora = jogo["Away"]

    # ---------------------------------
    # HISTÓRICO GERAL
    # ---------------------------------

    hist_casa = historico[casa]
    hist_fora = historico[fora]

    # Médias gerais
    home_gols_media = gols_media(hist_casa)
    home_gols_sofridos = gols_sofridos_media(hist_casa)
    home_aproveitamento = aproveitamento_historico(hist_casa)

    away_gols_media = gols_media(hist_fora)
    away_gols_sofridos = gols_sofridos_media(hist_fora)
    away_aproveitamento = aproveitamento_historico(hist_fora)

    # ---------------------------------
    # HISTÓRICO CASA/FORA
    # ---------------------------------

    hist_casa_mandante = historico_casa[casa]
    hist_fora_visitante = historico_fora[fora]

    home_gols_casa_media = media_gols_casa(hist_casa_mandante)
    home_gols_sofridos_casa = media_gols_sofridos_casa(
        hist_casa_mandante
    )
    home_aproveitamento_casa = aproveitamento_casa(
        hist_casa_mandante
    )

    away_gols_fora_media = media_gols_casa(
        hist_fora_visitante
    )
    away_gols_sofridos_fora = media_gols_sofridos_casa(
        hist_fora_visitante
    )
    away_aproveitamento_fora = aproveitamento_casa(
        hist_fora_visitante
    )

    # ---------------------------------
    # VITÓRIAS / DERROTAS
    # ---------------------------------

    home_vitorias = sum(1 for x in hist_casa if x[2] == 3)
    home_derrotas = sum(1 for x in hist_casa if x[2] == 0)

    away_vitorias = sum(1 for x in hist_fora if x[2] == 3)
    away_derrotas = sum(1 for x in hist_fora if x[2] == 0)

    # ---------------------------------
    # DIFERENÇAS
    # ---------------------------------

    diferenca_gols_media = (
        home_gols_media - away_gols_media
    )

    diferenca_gols_sofridos = (
        home_gols_sofridos - away_gols_sofridos
    )

    diferenca_aproveitamento = (
        home_aproveitamento - away_aproveitamento
    )

    diferenca_gols_casa_fora = (
        home_gols_casa_media - away_gols_fora_media
    )

    diferenca_aproveitamento_casa_fora = (
        home_aproveitamento_casa -
        away_aproveitamento_fora
    )

    diferenca_vitorias = (
        home_vitorias - away_vitorias
    )

    diferenca_derrotas = (
        home_derrotas - away_derrotas
    )

    # ---------------------------------
    # ÚLTIMOS 5 E 10 JOGOS
    # ---------------------------------

    home_pontos_5 = ultimos_pontos(
        [x[2] for x in hist_casa],
        5
    )

    away_pontos_5 = ultimos_pontos(
        [x[2] for x in hist_fora],
        5
    )

    home_pontos_10 = ultimos_pontos(
        [x[2] for x in hist_casa],
        10
    )

    away_pontos_10 = ultimos_pontos(
        [x[2] for x in hist_fora],
        10
    )

    home_empates_5 = sum(
        1 for x in hist_casa[-5:]
        if x[2] == 1
    )

    away_empates_5 = sum(
        1 for x in hist_fora[-5:]
        if x[2] == 1
    )

    home_empates_10 = sum(
        1 for x in hist_casa[-10:]
        if x[2] == 1
    )

    away_empates_10 = sum(
        1 for x in hist_fora[-10:]
        if x[2] == 1
    )

    # ---------------------------------
    # MÉDIAS DOS ÚLTIMOS 10
    # ---------------------------------

    home_gols_media_10 = gols_media(
        hist_casa,
        10
    )

    away_gols_media_10 = gols_media(
        hist_fora,
        10
    )

    home_gols_sofridos_media_10 = gols_sofridos_media(
        hist_casa,
        10
    )

    away_gols_sofridos_media_10 = gols_sofridos_media(
        hist_fora,
        10
    )

    home_gols_casa_media_10 = media_gols_casa(
        hist_casa_mandante,
        10
    )

    away_gols_fora_media_10 = media_gols_casa(
        hist_fora_visitante,
        10
    )

    home_aproveitamento_casa_10 = aproveitamento_casa(
        hist_casa_mandante,
        10
    )

    away_aproveitamento_fora_10 = aproveitamento_casa(
        hist_fora_visitante,
        10
    )

    # ---------------------------------
    # DIFERENÇAS ÚLTIMOS 5 / 10
    # ---------------------------------

    diferenca_pontos_5 = (
        home_pontos_5 - away_pontos_5
    )

    diferenca_pontos_10 = (
        home_pontos_10 - away_pontos_10
    )

    diferenca_empates_5 = (
        home_empates_5 - away_empates_5
    )

    diferenca_empates_10 = (
        home_empates_10 - away_empates_10
    )

    diferenca_gols_media_10 = (
        home_gols_media_10 -
        away_gols_media_10
    )

    diferenca_gols_sofridos_media_10 = (
        home_gols_sofridos_media_10 -
        away_gols_sofridos_media_10
    )

    # ---------------------------------
    # ELO
    # ---------------------------------

    elo_home = elo[casa]
    elo_away = elo[fora]

    diferenca_elo = elo_home - elo_away

    # ---------------------------------
    # SALVAR CARACTERÍSTICAS
    # ---------------------------------

    dados_processados.append({

        "Home_Gols_Media":
            home_gols_media,

        "Home_Gols_Sofridos_Media":
            home_gols_sofridos,

        "Home_Aproveitamento":
            home_aproveitamento,

        "Away_Gols_Media":
            away_gols_media,

        "Away_Gols_Sofridos_Media":
            away_gols_sofridos,

        "Away_Aproveitamento":
            away_aproveitamento,

        "Home_Gols_Casa_Media":
            home_gols_casa_media,

        "Home_Gols_Sofridos_Casa_Media":
            home_gols_sofridos_casa,

        "Home_Aproveitamento_Casa":
            home_aproveitamento_casa,

        "Away_Gols_Fora_Media":
            away_gols_fora_media,

        "Away_Gols_Sofridos_Fora_Media":
            away_gols_sofridos_fora,

        "Away_Aproveitamento_Fora":
            away_aproveitamento_fora,

        "Home_Vitorias":
            home_vitorias,

        "Home_Derrotas":
            home_derrotas,

        "Away_Vitorias":
            away_vitorias,

        "Away_Derrotas":
            away_derrotas,

        "Diferenca_Gols_Media":
            diferenca_gols_media,

        "Diferenca_Gols_Sofridos":
            diferenca_gols_sofridos,

        "Diferenca_Aproveitamento":
            diferenca_aproveitamento,

        "Diferenca_Gols_Casa_Fora":
            diferenca_gols_casa_fora,

        "Diferenca_Aproveitamento_Casa_Fora":
            diferenca_aproveitamento_casa_fora,

        "Diferenca_Vitorias":
            diferenca_vitorias,

        "Diferenca_Derrotas":
            diferenca_derrotas,

        "Home_Pontos_5":
            home_pontos_5,

        "Away_Pontos_5":
            away_pontos_5,

        "Home_Pontos_10":
            home_pontos_10,

        "Away_Pontos_10":
            away_pontos_10,

        "Home_Empates_5":
            home_empates_5,

        "Away_Empates_5":
            away_empates_5,

        "Home_Empates_10":
            home_empates_10,

        "Away_Empates_10":
            away_empates_10,

        "Home_Gols_Media_10":
            home_gols_media_10,

        "Away_Gols_Media_10":
            away_gols_media_10,

        "Home_Gols_Sofridos_Media_10":
            home_gols_sofridos_media_10,

        "Away_Gols_Sofridos_Media_10":
            away_gols_sofridos_media_10,

        "Home_Gols_Casa_Media_10":
            home_gols_casa_media_10,

        "Away_Gols_Fora_Media_10":
            away_gols_fora_media_10,

        "Home_Aproveitamento_Casa_10":
            home_aproveitamento_casa_10,

        "Away_Aproveitamento_Fora_10":
            away_aproveitamento_fora_10,

        "Diferenca_Pontos_5":
            diferenca_pontos_5,

        "Diferenca_Pontos_10":
            diferenca_pontos_10,

        "Diferenca_Empates_5":
            diferenca_empates_5,

        "Diferenca_Empates_10":
            diferenca_empates_10,

        "Diferenca_Gols_Media_10":
            diferenca_gols_media_10,

        "Diferenca_Gols_Sofridos_Media_10":
            diferenca_gols_sofridos_media_10,

        "Elo_Home":
            elo_home,

        "Elo_Away":
            elo_away,

        "Diferenca_Elo":
            diferenca_elo,

        "Home":
            casa,

        "Away":
            fora,

        "Date":
            jogo["Date"],

        "Res":
            jogo["Res"]
    })

    # ---------------------------------
    # ATUALIZAR HISTÓRICO
    # ---------------------------------

    gols_casa = int(jogo["HG"])
    gols_fora = int(jogo["AG"])
    resultado = jogo["Res"]

    if resultado == "H":
        pontos_casa = 3
        pontos_fora = 0
    elif resultado == "D":
        pontos_casa = 1
        pontos_fora = 1
    else:
        pontos_casa = 0
        pontos_fora = 3

    historico[casa].append(
        (gols_casa, gols_fora, pontos_casa)
    )

    historico[fora].append(
        (gols_fora, gols_casa, pontos_fora)
    )

    historico_casa[casa].append(
        (gols_casa, gols_fora, pontos_casa)
    )

    historico_fora[fora].append(
        (gols_fora, gols_casa, pontos_fora)
    )

    # Atualizar Elo somente DEPOIS da partida
    novo_elo_casa, novo_elo_fora = atualizar_elo(
        elo_home,
        elo_away,
        resultado
    )

    elo[casa] = novo_elo_casa
    elo[fora] = novo_elo_fora


# =========================
# SALVAR DATASET
# =========================

df_processado = pd.DataFrame(dados_processados)

df_processado.to_csv(
    ARQUIVO_SAIDA,
    index=False
)

print()
print("Dataset processado com sucesso!")
print(f"Partidas processadas: {len(df_processado)}")
print(
    f"Total de características criadas: "
    f"{len(df_processado.columns) - 4}"
)

print()
print("Novas características adicionadas:")
print("- Elo_Home")
print("- Elo_Away")
print("- Diferenca_Elo")

print()
print(f"Arquivo salvo em: {ARQUIVO_SAIDA}")