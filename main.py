import pandas as pd

dados = pd.read_csv("dataset/BRA.csv")

# Remover partidas sem resultado
dados = dados.dropna(subset=["HG", "AG", "Res"])

# Converter a data
dados["Date"] = pd.to_datetime(dados["Date"], dayfirst=True)

# Ordenar as partidas por data
dados = dados.sort_values("Date").reset_index(drop=True)

# Histórico dos times
historico = {}

# Listas para guardar as novas informações
home_gols_media = []
home_gols_sofridos_media = []
home_aproveitamento = []

away_gols_media = []
away_gols_sofridos_media = []
away_aproveitamento = []


# Analisar cada partida
for _, jogo in dados.iterrows():

    time_casa = jogo["Home"]
    time_fora = jogo["Away"]

    # Histórico dos dois times
    historico_casa = historico.get(time_casa, [])
    historico_fora = historico.get(time_fora, [])

    # Somente os últimos 5 jogos
    ultimos_casa = historico_casa[-5:]
    ultimos_fora = historico_fora[-5:]


    # =========================
    # TIME DA CASA
    # =========================

    if len(ultimos_casa) > 0:

        media_gols_casa = sum(
            jogo_info["marcados"] for jogo_info in ultimos_casa
        ) / len(ultimos_casa)

        media_sofridos_casa = sum(
            jogo_info["sofridos"] for jogo_info in ultimos_casa
        ) / len(ultimos_casa)

        pontos_casa = sum(
            jogo_info["pontos"] for jogo_info in ultimos_casa
        )

        aproveitamento_casa = (
            pontos_casa / (len(ultimos_casa) * 3)
        ) * 100

    else:

        media_gols_casa = 0
        media_sofridos_casa = 0
        aproveitamento_casa = 0


    # =========================
    # TIME VISITANTE
    # =========================

    if len(ultimos_fora) > 0:

        media_gols_fora = sum(
            jogo_info["marcados"] for jogo_info in ultimos_fora
        ) / len(ultimos_fora)

        media_sofridos_fora = sum(
            jogo_info["sofridos"] for jogo_info in ultimos_fora
        ) / len(ultimos_fora)

        pontos_fora = sum(
            jogo_info["pontos"] for jogo_info in ultimos_fora
        )

        aproveitamento_fora = (
            pontos_fora / (len(ultimos_fora) * 3)
        ) * 100

    else:

        media_gols_fora = 0
        media_sofridos_fora = 0
        aproveitamento_fora = 0


    # Guardar as informações
    home_gols_media.append(media_gols_casa)
    home_gols_sofridos_media.append(media_sofridos_casa)
    home_aproveitamento.append(aproveitamento_casa)

    away_gols_media.append(media_gols_fora)
    away_gols_sofridos_media.append(media_sofridos_fora)
    away_aproveitamento.append(aproveitamento_fora)


    # =========================
    # RESULTADO DA PARTIDA
    # =========================

    if jogo["Res"] == "H":

        pontos_casa = 3
        pontos_fora = 0

    elif jogo["Res"] == "D":

        pontos_casa = 1
        pontos_fora = 1

    else:

        pontos_casa = 0
        pontos_fora = 3


    # =========================
    # ADICIONAR PARTIDA AO HISTÓRICO
    # =========================

    historico.setdefault(time_casa, []).append({
        "marcados": jogo["HG"],
        "sofridos": jogo["AG"],
        "pontos": pontos_casa
    })

    historico.setdefault(time_fora, []).append({
        "marcados": jogo["AG"],
        "sofridos": jogo["HG"],
        "pontos": pontos_fora
    })


# Criar as novas colunas
dados["Home_Gols_Media"] = home_gols_media
dados["Home_Gols_Sofridos_Media"] = home_gols_sofridos_media
dados["Home_Aproveitamento"] = home_aproveitamento

dados["Away_Gols_Media"] = away_gols_media
dados["Away_Gols_Sofridos_Media"] = away_gols_sofridos_media
dados["Away_Aproveitamento"] = away_aproveitamento


# Mostrar o resultado
print(dados[[
    "Date",
    "Home",
    "Away",
    "Home_Gols_Media",
    "Home_Gols_Sofridos_Media",
    "Home_Aproveitamento",
    "Away_Gols_Media",
    "Away_Gols_Sofridos_Media",
    "Away_Aproveitamento",
    "Res"
]].head(20))

dados.to_csv("dataset/BRA_processado.csv", index=False)

print("\nDataset processado salvo com sucesso!")