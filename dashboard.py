import streamlit as st
import pandas as pd
import numpy as np

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

    .match-teams {
        font-size: 27px;
        font-weight: 800;
    }

    .vs {
        color: #888;
        padding: 0 10px;
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

aba1, aba2, aba3 = st.tabs([
    "📊 Visão geral",
    "🔮 Previsão",
    "📈 Análise"
])


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with aba1:

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

with aba2:

    st.header("🔮 Previsão de partida")

    st.caption(
        "Escolha os times para estimar xG e probabilidades."
    )

    times = sorted(
        set(dados["Home"])
        |
        set(dados["Away"])
    )

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

        placar = (
            placar_mais_provavel(
                xg_home,
                xg_away
            )
        )

        st.divider()

        st.markdown(
            f"""
            <div class="match-card">
                <div class="match-teams">
                    {time_casa}
                    <span class="vs">×</span>
                    {time_fora}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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

        if previsao == "Casa":

            st.success(
                f"🏆 Previsão: **vitória do {time_casa}**"
            )

        elif previsao == "Fora":

            st.success(
                f"🏆 Previsão: **vitória do {time_fora}**"
            )

        else:

            st.info(
                "🤝 Previsão: **empate**"
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

        grafico = pd.DataFrame({

            "Resultado": nomes,

            "Probabilidade":
                probabilidades
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
                "Placar mais provável",
                f"{placar[0]} × {placar[1]}"
            )

        with col2:

            st.metric(
                "Gols esperados",
                f"{xg_home + xg_away:.2f}"
            )

        st.divider()

        st.subheader(
            "5 placares mais prováveis"
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

with aba3:

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
