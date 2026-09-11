import pandas as pd


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

dados = pd.read_csv("dataset/BRA.csv")

print("Dataset carregado com sucesso!")


# ============================================================
# 2. LIMPEZA DOS DADOS
# ============================================================

dados = dados.dropna(subset=["HG", "AG", "Res"])

dados["Date"] = pd.to_datetime(
    dados["Date"],
    dayfirst=True
)

dados = dados.sort_values("Date").reset_index(drop=True)


# ============================================================
# 3. CONFIGURAÇÃO DO ELO
# ============================================================

ELO_INICIAL = 1500
K = 20

# Vantagem do mando de campo.
# O mandante recebe esse valor antes do cálculo
# da expectativa de resultado.
VANTAGEM_CASA = 60


# ============================================================
# 4. ESTRUTURAS DE HISTÓRICO
# ============================================================

historico = {}
historico_casa = {}
ultimos_casa = {}

# Elo atual de cada equipe
elo = {}


# ============================================================
# 5. FUNÇÕES AUXILIARES
# ============================================================

def obter_historico(time):
    if time not in historico:
        historico[time] = []
    return historico[time]


def obter_historico_casa(time):
    if time not in historico_casa:
        historico_casa[time] = []
    return historico_casa[time]


def obter_ultimos_casa(time):
    if time not in ultimos_casa:
        ultimos_casa[time] = []
    return ultimos_casa[time]


def obter_elo(time):
    if time not in elo:
        elo[time] = ELO_INICIAL

    return elo[time]


def calcular_media(lista, chave, quantidade=None):
    if not lista:
        return 0

    if quantidade is not None:
        lista = lista[-quantidade:]

    if not lista:
        return 0

    return sum(item[chave] for item in lista) / len(lista)


def calcular_aproveitamento(lista, quantidade=None):
    if not lista:
        return 0

    if quantidade is not None:
        lista = lista[-quantidade:]

    if not lista:
        return 0

    pontos = sum(item["pontos"] for item in lista)

    return pontos / (len(lista) * 3)


def calcular_vitorias(lista):
    return sum(
        1 for item in lista
        if item["resultado"] == "V"
    )


def calcular_derrotas(lista):
    return sum(
        1 for item in lista
        if item["resultado"] == "D"
    )


def calcular_empates(lista, quantidade=None):
    if quantidade is not None:
        lista = lista[-quantidade:]

    return sum(
        1 for item in lista
        if item["resultado"] == "E"
    )


# ============================================================
# 6. FUNÇÕES DO ELO
# ============================================================

def expectativa_elo(elo_casa, elo_fora):
    """
    Calcula a expectativa de vitória do mandante.

    Quanto maior a diferença de Elo, maior a expectativa
    de vitória do time mais forte.
    """

    diferenca = (
        elo_fora - (elo_casa + VANTAGEM_CASA)
    )

    expectativa = 1 / (
        1 + 10 ** (diferenca / 400)
    )

    return expectativa


def atualizar_elo(
    elo_casa,
    elo_fora,
    resultado_casa
):
    """
    Atualiza o Elo depois da partida.

    resultado_casa:
        1.0 = vitória do mandante
        0.5 = empate
        0.0 = derrota do mandante
    """

    expectativa_casa = expectativa_elo(
        elo_casa,
        elo_fora
    )

    expectativa_fora = 1 - expectativa_casa

    novo_elo_casa = (
        elo_casa
        + K * (resultado_casa - expectativa_casa)
    )

    novo_elo_fora = (
        elo_fora
        + K * (
            (1 - resultado_casa)
            - expectativa_fora
        )
    )

    return novo_elo_casa, novo_elo_fora


# ============================================================
# 7. PROCESSAR CADA PARTIDA
# ============================================================

dados_processados = []


for _, linha in dados.iterrows():

    home = linha["Home"]
    away = linha["Away"]

    gols_home = float(linha["HG"])
    gols_away = float(linha["AG"])

    resultado = linha["Res"]


    # --------------------------------------------------------
    # HISTÓRICO DOS TIMES
    # --------------------------------------------------------

    h_home = obter_historico(home)
    h_away = obter_historico(away)

    hc_home = obter_historico_casa(home)
    hf_away = obter_ultimos_casa(away)


    # --------------------------------------------------------
    # ELO ANTES DA PARTIDA
    # --------------------------------------------------------
    #
    # MUITO IMPORTANTE:
    #
    # Pegamos o Elo ANTES de atualizar com o resultado
    # dessa partida.
    #
    # Isso evita vazamento de dados.
    # --------------------------------------------------------

    elo_home = obter_elo(home)
    elo_away = obter_elo(away)

    diferenca_elo = (
        (elo_home + VANTAGEM_CASA)
        - elo_away
    )


    # ========================================================
    # FEATURES GERAIS
    # ========================================================

    home_gols_media = calcular_media(
        h_home,
        "gols_marcados"
    )

    home_gols_sofridos_media = calcular_media(
        h_home,
        "gols_sofridos"
    )

    home_aproveitamento = calcular_aproveitamento(
        h_home
    )


    away_gols_media = calcular_media(
        h_away,
        "gols_marcados"
    )

    away_gols_sofridos_media = calcular_media(
        h_away,
        "gols_sofridos"
    )

    away_aproveitamento = calcular_aproveitamento(
        h_away
    )


    # ========================================================
    # FEATURES CASA / FORA
    # ========================================================

    home_gols_casa_media = calcular_media(
        hc_home,
        "gols_marcados"
    )

    home_gols_sofridos_casa_media = calcular_media(
        hc_home,
        "gols_sofridos"
    )

    home_aproveitamento_casa = calcular_aproveitamento(
        hc_home
    )


    away_gols_fora_media = calcular_media(
        hf_away,
        "gols_marcados"
    )

    away_gols_sofridos_fora_media = calcular_media(
        hf_away,
        "gols_sofridos"
    )

    away_aproveitamento_fora = calcular_aproveitamento(
        hf_away
    )


    # ========================================================
    # VITÓRIAS E DERROTAS
    # ========================================================

    home_vitorias = calcular_vitorias(h_home)
    home_derrotas = calcular_derrotas(h_home)

    away_vitorias = calcular_vitorias(h_away)
    away_derrotas = calcular_derrotas(h_away)


    # ========================================================
    # DIFERENÇAS
    # ========================================================

    diferenca_gols_media = (
        home_gols_media
        - away_gols_media
    )

    diferenca_gols_sofridos = (
        home_gols_sofridos_media
        - away_gols_sofridos_media
    )

    diferenca_aproveitamento = (
        home_aproveitamento
        - away_aproveitamento
    )

    diferenca_gols_casa_fora = (
        home_gols_casa_media
        - away_gols_fora_media
    )

    diferenca_aproveitamento_casa_fora = (
        home_aproveitamento_casa
        - away_aproveitamento_fora
    )

    diferenca_vitorias = (
        home_vitorias
        - away_vitorias
    )

    diferenca_derrotas = (
        home_derrotas
        - away_derrotas
    )


    # ========================================================
    # ÚLTIMOS 5 E 10 JOGOS
    # ========================================================

    home_5 = h_home[-5:]
    away_5 = h_away[-5:]

    home_10 = h_home[-10:]
    away_10 = h_away[-10:]


    home_pontos_5 = sum(
        item["pontos"]
        for item in home_5
    )

    away_pontos_5 = sum(
        item["pontos"]
        for item in away_5
    )


    home_pontos_10 = sum(
        item["pontos"]
        for item in home_10
    )

    away_pontos_10 = sum(
        item["pontos"]
        for item in away_10
    )


    home_empates_5 = calcular_empates(
        h_home,
        5
    )

    away_empates_5 = calcular_empates(
        h_away,
        5
    )


    home_empates_10 = calcular_empates(
        h_home,
        10
    )

    away_empates_10 = calcular_empates(
        h_away,
        10
    )


    # ========================================================
    # MÉDIAS DOS ÚLTIMOS 10
    # ========================================================

    home_gols_media_10 = calcular_media(
        h_home,
        "gols_marcados",
        10
    )

    away_gols_media_10 = calcular_media(
        h_away,
        "gols_marcados",
        10
    )


    home_gols_sofridos_media_10 = calcular_media(
        h_home,
        "gols_sofridos",
        10
    )

    away_gols_sofridos_media_10 = calcular_media(
        h_away,
        "gols_sofridos",
        10
    )


    # ========================================================
    # CASA / FORA NOS ÚLTIMOS 10
    # ========================================================

    home_gols_casa_media_10 = calcular_media(
        hc_home,
        "gols_marcados",
        10
    )

    away_gols_fora_media_10 = calcular_media(
        hf_away,
        "gols_marcados",
        10
    )


    home_aproveitamento_casa_10 = (
        calcular_aproveitamento(
            hc_home,
            10
        )
    )

    away_aproveitamento_fora_10 = (
        calcular_aproveitamento(
            hf_away,
            10
        )
    )


    # ========================================================
    # DIFERENÇAS DOS ÚLTIMOS 5 E 10
    # ========================================================

    diferenca_pontos_5 = (
        home_pontos_5
        - away_pontos_5
    )

    diferenca_pontos_10 = (
        home_pontos_10
        - away_pontos_10
    )


    diferenca_empates_5 = (
        home_empates_5
        - away_empates_5
    )

    diferenca_empates_10 = (
        home_empates_10
        - away_empates_10
    )


    diferenca_gols_media_10 = (
        home_gols_media_10
        - away_gols_media_10
    )

    diferenca_gols_sofridos_media_10 = (
        home_gols_sofridos_media_10
        - away_gols_sofridos_media_10
    )


    # ========================================================
    # SALVAR FEATURES DA PARTIDA
    # ========================================================

    dados_processados.append({

        # ----------------------------------------------------
        # Features originais
        # ----------------------------------------------------

        "Home_Gols_Media":
            home_gols_media,

        "Home_Gols_Sofridos_Media":
            home_gols_sofridos_media,

        "Home_Aproveitamento":
            home_aproveitamento,


        "Away_Gols_Media":
            away_gols_media,

        "Away_Gols_Sofridos_Media":
            away_gols_sofridos_media,

        "Away_Aproveitamento":
            away_aproveitamento,


        "Home_Gols_Casa_Media":
            home_gols_casa_media,

        "Home_Gols_Sofridos_Casa_Media":
            home_gols_sofridos_casa_media,

        "Home_Aproveitamento_Casa":
            home_aproveitamento_casa,


        "Away_Gols_Fora_Media":
            away_gols_fora_media,

        "Away_Gols_Sofridos_Fora_Media":
            away_gols_sofridos_fora_media,

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


        # ----------------------------------------------------
        # Últimos 5 / 10
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ELO
        # ----------------------------------------------------

        "Elo_Home":
            elo_home,

        "Elo_Away":
            elo_away,

        "Diferenca_Elo":
            diferenca_elo,


        # ----------------------------------------------------
        # Informações da partida
        # ----------------------------------------------------

        "Home":
            home,

        "Away":
            away,

        "Date":
            linha["Date"],

        "Res":
            resultado
    })


    # ========================================================
    # 8. ATUALIZAR ELO APÓS A PARTIDA
    # ========================================================

    if resultado == "H":

        resultado_elo = 1.0

    elif resultado == "D":

        resultado_elo = 0.5

    else:

        resultado_elo = 0.0


    novo_elo_home, novo_elo_away = atualizar_elo(
        elo_home,
        elo_away,
        resultado_elo
    )


    elo[home] = novo_elo_home
    elo[away] = novo_elo_away


    # ========================================================
    # 9. ATUALIZAR HISTÓRICO
    # ========================================================

    if gols_home > gols_away:

        pontos_home = 3
        pontos_away = 0

        resultado_home = "V"
        resultado_away = "D"

    elif gols_home < gols_away:

        pontos_home = 0
        pontos_away = 3

        resultado_home = "D"
        resultado_away = "V"

    else:

        pontos_home = 1
        pontos_away = 1

        resultado_home = "E"
        resultado_away = "E"


    # Histórico do mandante
    h_home.append({
        "gols_marcados":
            gols_home,

        "gols_sofridos":
            gols_away,

        "pontos":
            pontos_home,

        "resultado":
            resultado_home
    })


    # Histórico do visitante
    h_away.append({
        "gols_marcados":
            gols_away,

        "gols_sofridos":
            gols_home,

        "pontos":
            pontos_away,

        "resultado":
            resultado_away
    })


    # Histórico de partidas em casa
    hc_home.append({
        "gols_marcados":
            gols_home,

        "gols_sofridos":
            gols_away,

        "pontos":
            pontos_home,

        "resultado":
            resultado_home
    })


    # Histórico de partidas fora
    hf_away.append({
        "gols_marcados":
            gols_away,

        "gols_sofridos":
            gols_home,

        "pontos":
            pontos_away,

        "resultado":
            resultado_away
    })


# ============================================================
# 10. CRIAR DATAFRAME FINAL
# ============================================================

dados_processados = pd.DataFrame(
    dados_processados
)


# ============================================================
# 11. SALVAR DATASET PROCESSADO
# ============================================================

dados_processados.to_csv(
    "dataset/BRA_processado.csv",
    index=False
)


# ============================================================
# 12. RESULTADO
# ============================================================

quantidade_features = 48

print()
print("Dataset processado com sucesso!")
print(
    "Partidas processadas:",
    len(dados_processados)
)

print(
    "Total de características criadas:",
    quantidade_features
)

print()
print("Novas características adicionadas:")
print("- Elo_Home")
print("- Elo_Away")
print("- Diferenca_Elo")

print()
print(
    "Arquivo salvo em: "
    "dataset/BRA_processado.csv"
)