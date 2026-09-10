import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Carregar o dataset processado
dados = pd.read_csv("dataset/BRA_processado.csv")


# Características que o modelo vai analisar
caracteristicas = [
    "Home_Gols_Media",
    "Home_Gols_Sofridos_Media",
    "Home_Aproveitamento",
    "Away_Gols_Media",
    "Away_Gols_Sofridos_Media",
    "Away_Aproveitamento"
]


# X = informações usadas para fazer a previsão
X = dados[caracteristicas]

# y = resultado que queremos prever
y = dados["Res"]


# Separar os dados:
# 80% para treinamento
# 20% para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Criar o modelo
modelo = LogisticRegression(max_iter=1000)


# Ensinar o modelo
modelo.fit(X_treino, y_treino)


# Fazer previsões
previsoes = modelo.predict(X_teste)


# Calcular a precisão
precisao = accuracy_score(y_teste, previsoes)


print("================================")
print("       DATAPREDICT")
print("================================")

print(f"\nPrecisao do modelo: {precisao * 100:.2f}%")

matriz = confusion_matrix(y_teste, previsoes, labels=["H", "D", "A"])

print("\nMatriz de confusao:")
print(matriz)