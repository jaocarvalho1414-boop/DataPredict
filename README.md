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

## Evolução do projeto

O DataPredict passou por várias etapas de desenvolvimento, nas quais o modelo foi sendo testado, avaliado e aprimorado.

### V1 — Primeiro modelo

Foi desenvolvido um primeiro modelo de classificação utilizando características históricas das equipes, como médias de gols, aproveitamento e desempenho recente.

Foram testados diferentes algoritmos de Machine Learning, incluindo:

* Logistic Regression
* Random Forest
* HistGradientBoosting

O primeiro objetivo foi estabelecer uma base para comparar os modelos e entender o comportamento das previsões.

### V2 — Separação entre treino, validação e teste

O projeto passou a utilizar uma divisão cronológica de:

* 60% para treinamento
* 20% para validação
* 20% para teste

Essa abordagem foi adotada para respeitar a ordem temporal das partidas e evitar utilizar informações futuras durante o treinamento.

### V3 — Ajuste do modelo

Foram testados diferentes parâmetros do HistGradientBoosting, buscando melhorar o equilíbrio entre as classes:

* Vitória do mandante
* Empate
* Vitória do visitante

Também foi analisada a matriz de confusão para identificar onde o modelo apresentava maior dificuldade.

### V4 — Tratamento das classes

Foram realizados testes com diferentes pesos para as classes, principalmente para verificar se era possível melhorar a identificação de empates e vitórias do visitante.

Os testes mostraram que simplesmente aumentar o peso de determinadas classes não necessariamente melhorava o desempenho geral.

### V5 — Análise das probabilidades

O projeto passou a trabalhar não apenas com a classe prevista, mas também com as probabilidades de cada resultado.

Foram testados ajustes nas probabilidades para verificar se o modelo poderia representar melhor situações de equilíbrio entre as equipes.

### V6 — Modelo baseado em gols esperados

Foi adicionada uma abordagem baseada em distribuição de Poisson.

A partir das médias históricas de gols marcados e sofridos pelas equipes, o sistema passou a estimar:

* Gols esperados do mandante
* Gols esperados do visitante
* Probabilidade de vitória do mandante
* Probabilidade de empate
* Probabilidade de vitória do visitante

Também foi corrigido o mapeamento das classes de probabilidade do modelo de Machine Learning.

### V7 — Teste de ajuste para empates

Foi testado um mecanismo para aumentar a probabilidade de empate quando os gols esperados das equipes estavam muito próximos.

Apesar de aumentar a quantidade de empates previstos, o teste não apresentou melhora no conjunto de teste.

Por isso, o ajuste artificial foi descartado.

### V8 — Abordagem atual

A versão atual combina duas ideias:

1. **Gols esperados**, calculados a partir do desempenho ofensivo e defensivo das equipes.
2. **Contexto das equipes**, utilizando informações como forma recente, aproveitamento e Elo.

Os gols esperados são utilizados em um modelo de Poisson para estimar diferentes placares possíveis. A partir desses placares, são calculadas as probabilidades de vitória, empate e derrota.

O modelo de Machine Learning é utilizado como uma camada adicional de contexto, em vez de determinar sozinho o resultado.

O objetivo dessa evolução é tornar o DataPredict mais próximo de uma análise real de futebol, utilizando dados históricos para estimar o comportamento provável de uma partida.


## Desenvolvedor

João Pedro Vilela de Carvalho