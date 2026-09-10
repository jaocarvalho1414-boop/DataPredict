# DataPredict

Projeto de Machine Learning para análise e previsão de resultados de partidas de futebol.

## Sobre o projeto

O DataPredict utiliza dados históricos do Campeonato Brasileiro para analisar o desempenho recente das equipes e tentar prever o resultado de uma partida.

O projeto está sendo desenvolvido como uma forma de estudar Python, análise de dados e Machine Learning na prática.

## Dados utilizados

O projeto utiliza dados históricos de partidas do Campeonato Brasileiro.

As informações são utilizadas para criar características como:

- Média de gols marcados nos últimos 5 jogos
- Média de gols sofridos nos últimos 5 jogos
- Aproveitamento nos últimos 5 jogos

## Machine Learning

O primeiro modelo utilizado foi o **Logistic Regression**.

Os dados foram divididos em:

- 80% para treinamento
- 20% para teste

### Primeiro resultado

**Acurácia: 48,16%**

O primeiro modelo apresentou dificuldade para identificar empates, não realizando previsões da classe `D` no conjunto de teste.

Esse resultado será utilizado como base para testar outros modelos e melhorar o desempenho do projeto.

## Tecnologias

- Python
- Pandas
- Scikit-learn
- Machine Learning

## Próximos passos

- Analisar a distribuição dos resultados
- Melhorar as características utilizadas pelo modelo
- Testar Decision Tree
- Testar Random Forest
- Comparar os modelos
- Criar uma interface para realizar previsões

## Desenvolvedor

João Pedro Vilela de Carvalho