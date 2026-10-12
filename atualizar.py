import pandas as pd

URL = "https://www.football-data.co.uk/new/BRA.csv"
CAMINHO = "dataset/BRA.csv"
CHAVE = ["Date", "Home", "Away"]

def atualizar_banco():
    antigo = pd.read_csv(CAMINHO)
    novo = pd.read_csv(URL)

    combinado = pd.concat([antigo, novo], ignore_index=True)
    combinado = combinado.drop_duplicates(subset=CHAVE, keep="last")

    adicionados = len(combinado) - len(antigo)
    combinado.to_csv(CAMINHO, index=False)
    return adicionados

if __name__ == "__main__":
    print(f"{atualizar_banco()} partidas novas")