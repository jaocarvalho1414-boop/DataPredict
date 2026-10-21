import streamlit as st
import pandas as pd
import numpy as np
import requests
from html import escape

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from math import factorial, exp


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="DataPredict | Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ARQUIVO_PROCESSADO = "dataset/BRA_processado.csv"
ARQUIVO_ORIGINAL = "dataset/BRA.csv"

# Páginas usadas para buscar os escudos na Wikipédia em português.
# Se a imagem não estiver disponível, o dashboard mostra as iniciais do clube.
PAGINAS_ESCUDOS = {
    "Athletico-PR": "Club Athletico Paranaense",
    "Atletico-MG": "Clube Atlético Mineiro",
    "Bahia": "Esporte Clube Bahia",
    "Botafogo RJ": "Botafogo de Futebol e Regatas",
    "Chapecoense-SC": "Associação Chapecoense de Futebol",
    "Corinthians": "Sport Club Corinthians Paulista",
    "Coritiba": "Coritiba Foot Ball Club",
    "Cruzeiro": "Cruzeiro Esporte Clube",
    "Flamengo RJ": "Clube de Regatas do Flamengo",
    "Fluminense": "Fluminense Football Club",
    "Gremio": "Grêmio Foot-Ball Porto Alegrense",
    "Internacional": "Sport Club Internacional",
    "Mirassol": "Mirassol Futebol Clube",
    "Palmeiras": "Sociedade Esportiva Palmeiras",
    "Bragantino": "Red Bull Bragantino",
    "Remo": "Clube do Remo",
    "Santos": "Santos Futebol Clube",
    "Sao Paulo": "São Paulo Futebol Clube",
    "Vasco": "Club de Regatas Vasco da Gama",
    "Vitoria": "Esporte Clube Vitória",
}


@st.cache_data(ttl=604800, show_spinner=False)
def buscar_escudo(time):
    """Busca e guarda em cache a miniatura do escudo na Wikipédia."""
    pagina = PAGINAS_ESCUDOS.get(time)
    if not pagina:
        return None

    try:
        resposta = requests.get(
            "https://pt.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 300,
                "titles": pagina,
            },
            headers={"User-Agent": "DataPredictDashboard/1.0 (club crest display)"},
            timeout=5,
        )
        resposta.raise_for_status()
        paginas = resposta.json().get("query", {}).get("pages", {})
        for item in paginas.values():
            miniatura = item.get("thumbnail", {}).get("source")
            if miniatura:
                return miniatura
    except (requests.RequestException, ValueError, TypeError):
        pass

    return None


def html_escudo(time):
    """Gera o HTML do escudo; usa iniciais como alternativa se falhar."""
    url = buscar_escudo(time)
    nome_seguro = escape(time)

    if url:
        return (
            f'<img class="team-crest" src="{escape(url, quote=True)}" '
            f'alt="Escudo do {nome_seguro}" loading="lazy">'
        )

    iniciais = "".join(parte[0] for parte in time.replace("-", " ").split() if parte)[:3]
    return f'<div class="team-fallback" aria-label="{nome_seguro}">{escape(iniciais.upper())}</div>'


# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px;
        margin-bottom: 24px;
        background: linear-gradient(
            135deg,
            rgba(128,128,128,.10),
            rgba(128,128,128,.03)
        );
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #888;
        margin-top: 5px;
    }

    .match-card {
        text-align: center;
        padding: 22px;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px;
        margin: 10px 0 20px 0;
    }

    .match-teams-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: clamp(14px, 4vw, 48px);
        flex-wrap: wrap;
    }

    .team-side {
        display: flex;
        flex: 1 1 180px;
        min-width: 130px;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }

    .team-crest {
        width: 104px;
        height: 104px;
        object-fit: contain;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,.12));
    }

    .team-fallback {
        width: 104px;
        height: 104px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: 1px solid rgba(128,128,128,.35);
        background: rgba(128,128,128,.10);
        font-size: 25px;
        font-weight: 800;
        color: #888;
    }

    .match-team-name {
        font-size: clamp(17px, 2vw, 24px);
        line-height: 1.2;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .prediction-score-divider {
        width: 100%;
        height: 1px;
        background: rgba(128,128,128,.22);
        margin: 24px 0 18px 0;
    }

    .prediction-score-label {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.6px;
        color: #888;
        text-transform: uppercase;
    }

    .prediction-score {
        font-size: clamp(34px, 5vw, 48px);
        font-weight: 900;
        letter-spacing: 1px;
        line-height: 1.15;
        margin: 8px 0;
    }

    .prediction-outcome {
        font-size: 17px;
        font-weight: 700;
        margin-top: 6px;
    }

    .prediction-disclaimer {
        font-size: 12px;
        color: #888;
        margin-top: 10px;
    }

    .vs {
        color: #888;
        font-size: 23px;
        font-weight: 700;
        padding: 0 4px;
    }

    @media (max-width: 520px) {
        .team-crest, .team-fallback {
            width: 76px;
            height: 76px;
        }
        .team-side {
            min-width: 100px;
        }
    }

    .prediction {
        text-align: center;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.22);
    }

    .prediction-number {
        font-size: 30px;
        font-weight: 800;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        margin-top: 12px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 13px;
        padding: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CABEÇALHO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">⚽ DataPredict</div>
    <div class="hero-subtitle">
        Football Analytics & Prediction
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CARREGAMENTO
# ============================================================

@st.cache_data
def carregar_dados():

    df_proc = pd.read_csv(ARQUIVO_PROCESSADO)
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

    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)

    df_proc["Home"] = df_proc["Home"].astype(str)
    df_proc["Away"] = df_proc["Away"].astype(str)

    resultados = df[
        ["Date", "Home", "Away", "HG", "AG"]
    ].copy()

    dados = pd.merge(
        df_proc,
        resultados,
        on=["Date", "Home", "Away"],
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

    return (
        dados
        .sort_values("Date")
        .reset_index(drop=True)
    )


try:
    dados = carregar_dados()

except Exception as erro:

    st.error(
        "Não foi possível carregar os dados do projeto."
    )

    st.code(str(erro))

    st.stop()


# ============================================================
# FEATURES
# ============================================================

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

FEATURES = [
    coluna
    for coluna in FEATURES
    if coluna in dados.columns
]

X = (
    dados[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)

y = dados["Res"].astype(str).values

gols_home = dados["HG"].astype(float).values
gols_away = dados["AG"].astype(float).values


# ============================================================
# MODELOS xG
# ============================================================

@st.cache_resource
def treinar_modelos():

    limite = int(len(dados) * 0.80)

    modelo_home = make_pipeline(
        StandardScaler(),
        PoissonRegressor(
            alpha=0.5,
            max_iter=1000
        )
    )

    modelo_away = make_pipeline(
        StandardScaler(),
        PoissonRegressor(
            alpha=0.5,
            max_iter=1000
        )
    )

    modelo_home.fit(
        X.iloc[:limite],
        gols_home[:limite]
    )

    modelo_away.fit(
        X.iloc[:limite],
        gols_away[:limite]
    )

    return modelo_home, modelo_away


modelo_home, modelo_away = treinar_modelos()


# ============================================================
# FUNÇÕES DE PREVISÃO
# ============================================================

def probabilidades_xg(
    xg_home,
    xg_away
):

    diferenca = (
        xg_home - xg_away
    )

    distancia = abs(
        diferenca
    )

    empate = (
        0.20
        +
        0.20
        *
        np.exp(
            -1.40 * distancia
        )
    )

    empate = np.clip(
        empate,
        0.20,
        0.40
    )

    restante = (
        1.0 - empate
    )

    intensidade = 2.8

    vantagem_home = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -intensidade
                *
                diferenca
            )
        )
    )

    casa = (
        restante
        *
        vantagem_home
    )

    fora = (
        restante
        *
        (
            1.0
            -
            vantagem_home
        )
    )

    bonus_casa = 0.025

    casa += bonus_casa
    fora -= bonus_casa

    probabilidades = np.array([
        max(casa, 0.01),
        empate,
        max(fora, 0.01)
    ])

    return (
        probabilidades
        /
        probabilidades.sum()
    )


def construir_previsao(
    time_casa,
    time_fora
):

    jogos_casa = dados[
        (dados["Home"] == time_casa)
        |
        (dados["Away"] == time_casa)
    ].sort_values("Date")

    jogos_fora = dados[
        (dados["Home"] == time_fora)
        |
        (dados["Away"] == time_fora)
    ].sort_values("Date")

    if (
        jogos_casa.empty
        or jogos_fora.empty
    ):
        return None

    registro_casa = jogos_casa.iloc[-1]
    registro_fora = jogos_fora.iloc[-1]

    def estatistica_geral(
        registro,
        time,
        campo
    ):

        if registro["Home"] == time:

            return float(
                registro[
                    "Home_" + campo
                ]
            )

        return float(
            registro[
                "Away_" + campo
            ]
        )

    def estatistica_mando(
        registro,
        time,
        campo_casa,
        campo_fora
    ):

        if registro["Home"] == time:

            return float(
                registro[
                    campo_casa
                ]
            )

        return float(
            registro[
                campo_fora
            ]
        )

    linha = {

        "Home_Gols_Media":
            estatistica_geral(
                registro_casa,
                time_casa,
                "Gols_Media"
            ),

        "Home_Gols_Sofridos_Media":
            estatistica_geral(
                registro_casa,
                time_casa,
                "Gols_Sofridos_Media"
            ),

        "Home_Aproveitamento":
            estatistica_geral(
                registro_casa,
                time_casa,
                "Aproveitamento"
            ),

        "Away_Gols_Media":
            estatistica_geral(
                registro_fora,
                time_fora,
                "Gols_Media"
            ),

        "Away_Gols_Sofridos_Media":
            estatistica_geral(
                registro_fora,
                time_fora,
                "Gols_Sofridos_Media"
            ),

        "Away_Aproveitamento":
            estatistica_geral(
                registro_fora,
                time_fora,
                "Aproveitamento"
            ),

        "Home_Gols_Casa_Media":
            estatistica_mando(
                registro_casa,
                time_casa,
                "Home_Gols_Casa_Media",
                "Away_Gols_Fora_Media"
            ),

        "Home_Gols_Sofridos_Casa_Media":
            estatistica_mando(
                registro_casa,
                time_casa,
                "Home_Gols_Sofridos_Casa_Media",
                "Away_Gols_Sofridos_Fora_Media"
            ),

        "Home_Aproveitamento_Casa":
            estatistica_mando(
                registro_casa,
                time_casa,
                "Home_Aproveitamento_Casa",
                "Away_Aproveitamento_Fora"
            ),

        "Away_Gols_Fora_Media":
            estatistica_mando(
                registro_fora,
                time_fora,
                "Home_Gols_Casa_Media",
                "Away_Gols_Fora_Media"
            ),

        "Away_Gols_Sofridos_Fora_Media":
            estatistica_mando(
                registro_fora,
                time_fora,
                "Home_Gols_Sofridos_Casa_Media",
                "Away_Gols_Sofridos_Fora_Media"
            ),

        "Away_Aproveitamento_Fora":
            estatistica_mando(
                registro_fora,
                time_fora,
                "Home_Aproveitamento_Casa",
                "Away_Aproveitamento_Fora"
            ),

        "Home_Vitorias":
            estatistica_geral(
                registro_casa,
                time_casa,
                "Vitorias"
            ),

        "Home_Derrotas":
            estatistica_geral(
                registro_casa,
                time_casa,
                "Derrotas"
            ),

        "Away_Vitorias":
            estatistica_geral(
                registro_fora,
                time_fora,
                "Vitorias"
            ),

        "Away_Derrotas":
            estatistica_geral(
                registro_fora,
                time_fora,
                "Derrotas"
            )
    }

    linha[
        "Diferenca_Gols_Media"
    ] = (
        linha["Home_Gols_Media"]
        -
        linha["Away_Gols_Media"]
    )

    linha[
        "Diferenca_Gols_Sofridos"
    ] = (
        linha["Home_Gols_Sofridos_Media"]
        -
        linha["Away_Gols_Sofridos_Media"]
    )

    linha[
        "Diferenca_Aproveitamento"
    ] = (
        linha["Home_Aproveitamento"]
        -
        linha["Away_Aproveitamento"]
    )

    linha[
        "Diferenca_Gols_Casa_Fora"
    ] = (
        linha["Home_Gols_Casa_Media"]
        -
        linha["Away_Gols_Fora_Media"]
    )

    return (
        pd.DataFrame([linha])
        [FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


def placar_mais_provavel(
    xg_home,
    xg_away
):

    melhor = (
        0,
        0
    )

    maior = 0

    for gols_h in range(9):

        prob_h = (
            exp(-xg_home)
            *
            xg_home ** gols_h
            /
            factorial(gols_h)
        )

        for gols_a in range(9):

            prob_a = (
                exp(-xg_away)
                *
                xg_away ** gols_a
                /
                factorial(gols_a)
            )

            prob = (
                prob_h
                *
                prob_a
            )

            if prob > maior:

                maior = prob

                melhor = (
                    gols_h,
                    gols_a
                )

    return melhor


def top_placares(
    xg_home,
    xg_away,
    quantidade=5
):

    resultados = []

    for gols_h in range(7):

        prob_h = (
            exp(-xg_home)
            *
            xg_home ** gols_h
            /
            factorial(gols_h)
        )

        for gols_a in range(7):

            prob_a = (
                exp(-xg_away)
                *
                xg_away ** gols_a
                /
                factorial(gols_a)
            )

            resultados.append(
                (
                    f"{gols_h} × {gols_a}",
                    prob_h * prob_a
                )
            )

    resultados.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return resultados[:quantidade]


def placar_compativel_com_previsao(
    xg_home,
    xg_away,
    previsao
):
    """
    Retorna o placar individualmente mais provável que combine
    com o resultado 1X2 apontado pelo modelo.
    """
    resultados = []

    for gols_h in range(9):
        prob_h = (
            exp(-xg_home)
            * xg_home ** gols_h
            / factorial(gols_h)
        )

        for gols_a in range(9):
            if previsao == "Casa" and gols_h <= gols_a:
                continue
            if previsao == "Empate" and gols_h != gols_a:
                continue
            if previsao == "Fora" and gols_h >= gols_a:
                continue

            prob_a = (
                exp(-xg_away)
                * xg_away ** gols_a
                / factorial(gols_a)
            )

            resultados.append(
                (gols_h, gols_a, prob_h * prob_a)
            )

    if not resultados:
        return (0, 0)

    melhor = max(resultados, key=lambda item: item[2])
    return melhor[0], melhor[1]


def perfil_time(time):

    jogos = dados[
        (dados["Home"] == time)
        |
        (dados["Away"] == time)
    ].sort_values("Date").tail(10)

    if jogos.empty:
        return None

    gols_marcados = []
    gols_sofridos = []

    pontos = 0
    vitorias = 0
    empates = 0
    derrotas = 0

    for _, jogo in jogos.iterrows():

        if jogo["Home"] == time:

            gf = jogo["HG"]
            ga = jogo["AG"]
            resultado = jogo["Res"]

        else:

            gf = jogo["AG"]
            ga = jogo["HG"]

            resultado = {
                "H": "A",
                "D": "D",
                "A": "H"
            }[jogo["Res"]]

        gols_marcados.append(gf)
        gols_sofridos.append(ga)

        if resultado == "H":

            vitorias += 1
            pontos += 3

        elif resultado == "D":

            empates += 1
            pontos += 1

        else:

            derrotas += 1

    return {

        "pontos": pontos,
        "vitorias": vitorias,
        "empates": empates,
        "derrotas": derrotas,

        "gols_marcados":
            np.mean(gols_marcados),

        "gols_sofridos":
            np.mean(gols_sofridos)
    }


# ============================================================
# ABAS
# ============================================================

aba_previsao, aba_analise, aba_visao_geral = st.tabs([
    "🔮 Previsão",
    "📈 Análise",
    "📊 Visão geral"
])


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with aba_visao_geral:

    st.header("Visão geral")

    st.caption(
        "Uma visão geral dos dados utilizados pelo DataPredict."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Partidas analisadas",
            f"{len(dados):,}".replace(",", ".")
        )

    with col2:

        st.metric(
            "Características",
            len(FEATURES)
        )

    with col3:

        st.metric(
            "Vitórias da casa",
            f"{np.mean(y == 'H'):.1%}"
        )

    with col4:

        st.metric(
            "Empates",
            f"{np.mean(y == 'D'):.1%}"
        )

    st.divider()

    st.subheader(
        "Distribuição dos resultados"
    )

    distribuicao = pd.DataFrame({

        "Resultado":
            ["Casa", "Empate", "Fora"],

        "Partidas":
            [
                np.sum(y == "H"),
                np.sum(y == "D"),
                np.sum(y == "A")
            ]
    })

    st.bar_chart(
        distribuicao.set_index(
            "Resultado"
        )
    )

    st.divider()

    st.subheader(
        "Últimas partidas"
    )

    tabela = dados[
        [
            "Date",
            "Home",
            "Away",
            "HG",
            "AG",
            "Res"
        ]
    ].tail(20).copy()

    tabela["Resultado"] = tabela[
        "Res"
    ].map({
        "H": "Casa",
        "D": "Empate",
        "A": "Fora"
    })

    tabela = tabela.drop(
        columns=["Res"]
    )

    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABA 2 — PREVISÃO
# ============================================================

with aba_previsao:

    st.header("🔮 Previsão de partida")

    st.caption(
        "Escolha os times para estimar xG e probabilidades."
    )

     

    TIMES_SERIE_A_ATUAIS = [
        "Athletico-PR",
        "Atletico-MG",
        "Bahia",
        "Botafogo RJ",
        "Chapecoense-SC",
        "Corinthians",
        "Coritiba",
        "Cruzeiro",
        "Flamengo RJ",
        "Fluminense",
        "Gremio",
        "Internacional",
        "Mirassol",
        "Palmeiras",
        "Bragantino",
        "Remo",
        "Santos",
        "Sao Paulo",
        "Vasco",
        "Vitoria"
    ]

    times = [
        time
        for time in TIMES_SERIE_A_ATUAIS
        if time in dados["Home"].values
        or time in dados["Away"].values
    ]
    col1, col2 = st.columns(2)

    with col1:

        time_casa = st.selectbox(
            "🏠 Time da casa",
            times,
            index=(
                times.index("Palmeiras")
                if "Palmeiras" in times
                else 0
            )
        )

    with col2:

        visitantes = [
            time
            for time in times
            if time != time_casa
        ]

        time_fora = st.selectbox(
            "🚌 Time visitante",
            visitantes
        )

    analisar = st.button(
        "⚽ ANALISAR PARTIDA",
        use_container_width=True,
        type="primary"
    )

    if analisar:

        X_previsao = construir_previsao(
            time_casa,
            time_fora
        )

        if X_previsao is None:

            st.error(
                "Não existem dados suficientes."
            )

            st.stop()

        xg_home = float(
            np.clip(
                modelo_home.predict(
                    X_previsao
                )[0],
                0.05,
                5
            )
        )

        xg_away = float(
            np.clip(
                modelo_away.predict(
                    X_previsao
                )[0],
                0.05,
                5
            )
        )

        probabilidades = (
            probabilidades_xg(
                xg_home,
                xg_away
            )
        )

        st.divider()

        nomes = [
            "Casa",
            "Empate",
            "Fora"
        ]

        indice = int(
            np.argmax(
                probabilidades
            )
        )

        previsao = nomes[indice]

        # Escolhe o placar mais provável que combine com a tendência 1X2.
        placar = placar_compativel_com_previsao(
            xg_home,
            xg_away,
            previsao
        )

        if previsao == "Casa":
            resultado_texto = f"Vitória prevista: {time_casa}"
        elif previsao == "Fora":
            resultado_texto = f"Vitória prevista: {time_fora}"
        else:
            resultado_texto = "Empate previsto"

        escudo_casa = html_escudo(time_casa)
        escudo_fora = html_escudo(time_fora)

        st.markdown(
            f"""
            <div class="match-card">
                <div class="match-teams-row">
                    <div class="team-side">
                        {escudo_casa}
                        <div class="match-team-name">{escape(time_casa)}</div>
                    </div>
                    <div class="vs">×</div>
                    <div class="team-side">
                        {escudo_fora}
                        <div class="match-team-name">{escape(time_fora)}</div>
                    </div>
                </div>
                <div class="prediction-score-divider"></div>
                <div class="prediction-score-label">Placar previsto pela IA</div>
                <div class="prediction-score">{placar[0]} <span class="vs">×</span> {placar[1]}</div>
                <div class="prediction-outcome">{escape(resultado_texto)}</div>
                <div class="prediction-disclaimer">
                    Estimativa gerada por inteligência artificial. Não é o resultado real do jogo.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Leitura rápida do confronto
        diferenca_xg = xg_home - xg_away

        if abs(diferenca_xg) < 0.10:
            leitura = "O confronto aparece bastante equilibrado pelo xG."
        elif diferenca_xg > 0:
            leitura = (
                f"O xG indica vantagem ofensiva para **{time_casa}**, "
                "o que aumenta sua probabilidade de vitória."
            )
        else:
            leitura = (
                f"O xG indica vantagem ofensiva para **{time_fora}**, "
                "o que aumenta sua probabilidade de vitória."
            )

        st.markdown(
            f"""
            <div class="info-box">
                <strong>Leitura do modelo</strong><br>
                {leitura}<br>
                <span style="opacity:0.75">
                    A previsão combina a expectativa de gols (xG)
                    com a distribuição de resultados derivada dela.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader(
            "Probabilidades"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🏠 Casa",
                f"{probabilidades[0]:.1%}"
            )

        with col2:

            st.metric(
                "🤝 Empate",
                f"{probabilidades[1]:.1%}"
            )

        with col3:

            st.metric(
                "🚌 Fora",
                f"{probabilidades[2]:.1%}"
            )

        # Barras visuais de probabilidade
        for nome, prob in zip(nomes, probabilidades):
            st.progress(
                float(prob),
                text=f"{nome}: {prob:.1%}"
            )

        grafico = pd.DataFrame({
            "Resultado": nomes,
            "Probabilidade": probabilidades
        })

        st.bar_chart(
            grafico.set_index(
                "Resultado"
            )
        )

        st.divider()

        st.subheader(
            "Expectativa de gols"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "xG Casa",
                f"{xg_home:.2f}"
            )

        with col2:

            st.metric(
                "Diferença xG",
                f"{xg_home - xg_away:+.2f}"
            )

        with col3:

            st.metric(
                "xG Fora",
                f"{xg_away:.2f}"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Placar previsto pela IA",
                f"{placar[0]} × {placar[1]}"
            )

        with col2:

            st.metric(
                "Gols esperados",
                f"{xg_home + xg_away:.2f}"
            )

        st.divider()

        st.subheader(
            "5 placares com maior probabilidade individual"
        )

        placares = top_placares(
            xg_home,
            xg_away
        )

        tabela_placares = pd.DataFrame({

            "Placar":
                [
                    placar_nome
                    for placar_nome, _
                    in placares
                ],

            "Probabilidade":
                [
                    f"{prob:.1%}"
                    for _, prob
                    in placares
                ]
        })

        st.dataframe(
            tabela_placares,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "A lista mostra a probabilidade individual de cada placar. "
            "O placar em destaque é escolhido para ser compatível com a tendência "
            "de vitória, empate ou derrota indicada pelo modelo. Tudo é estimativa "
            "da IA e não representa o resultado real nem uma garantia."
        )

        st.divider()

        st.subheader(
            "Forma recente"
        )

        perfil_casa = perfil_time(
            time_casa
        )

        perfil_fora = perfil_time(
            time_fora
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"### 🏠 {time_casa}"
            )

            st.metric(
                "Pontos nos últimos 10",
                perfil_casa["pontos"]
            )

            st.write(
                f"**{perfil_casa['vitorias']}** "
                f"vitórias · "
                f"**{perfil_casa['empates']}** "
                f"empates · "
                f"**{perfil_casa['derrotas']}** "
                f"derrotas"
            )

            st.write(
                f"Gols marcados: "
                f"**{perfil_casa['gols_marcados']:.2f}** "
                f"por jogo"
            )

            st.write(
                f"Gols sofridos: "
                f"**{perfil_casa['gols_sofridos']:.2f}** "
                f"por jogo"
            )

        with col2:

            st.markdown(
                f"### 🚌 {time_fora}"
            )

            st.metric(
                "Pontos nos últimos 10",
                perfil_fora["pontos"]
            )

            st.write(
                f"**{perfil_fora['vitorias']}** "
                f"vitórias · "
                f"**{perfil_fora['empates']}** "
                f"empates · "
                f"**{perfil_fora['derrotas']}** "
                f"derrotas"
            )

            st.write(
                f"Gols marcados: "
                f"**{perfil_fora['gols_marcados']:.2f}** "
                f"por jogo"
            )

            st.write(
                f"Gols sofridos: "
                f"**{perfil_fora['gols_sofridos']:.2f}** "
                f"por jogo"
            )


# ============================================================
# ABA 3 — ANÁLISE
# ============================================================

with aba_analise:

    st.header("📈 Análise dos dados")

    st.caption(
        "Estatísticas gerais encontradas no dataset."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Média de gols da casa",
            f"{dados['HG'].mean():.2f}"
        )

    with col2:

        st.metric(
            "Média de gols fora",
            f"{dados['AG'].mean():.2f}"
        )

    with col3:

        st.metric(
            "Média total",
            f"{(dados['HG'] + dados['AG']).mean():.2f}"
        )

    st.divider()

    st.subheader(
        "Média de gols"
    )

    gols = pd.DataFrame({

        "Local":
            ["Casa", "Fora"],

        "Média":
            [
                dados["HG"].mean(),
                dados["AG"].mean()
            ]
    })

    st.bar_chart(
        gols.set_index(
            "Local"
        )
    )

    st.divider()

    st.subheader(
        "Resultados ao longo do tempo"
    )

    resultados_tempo = (
        dados
        .set_index("Date")
        .resample("YS")["Res"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(
            columns={
                "H": "Casa",
                "D": "Empate",
                "A": "Fora"
            }
        )
    )

    st.line_chart(
        resultados_tempo
    )

    st.divider()

    st.subheader(
        "Distribuição dos gols"
    )

    gols_distribuicao = pd.DataFrame({

        "Casa":
            dados["HG"].value_counts(),

        "Fora":
            dados["AG"].value_counts()
    }).fillna(0)

    gols_distribuicao.index = (
        gols_distribuicao.index
        .astype(int)
    )

    st.bar_chart(
        gols_distribuicao
    )

    st.divider()

    st.subheader(
        "Distribuição dos resultados"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Casa",
            f"{np.mean(y == 'H'):.1%}"
        )

    with col2:

        st.metric(
            "Empate",
            f"{np.mean(y == 'D'):.1%}"
        )

    with col3:

        st.metric(
            "Fora",
            f"{np.mean(y == 'A'):.1%}"
        )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "DataPredict — Football Analytics | "
    "Projeto desenvolvido por João Pedro Vilela de Carvalho"
)
