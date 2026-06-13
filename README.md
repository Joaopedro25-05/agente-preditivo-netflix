# Agente Preditivo Especialista — Netflix Movies and TV Shows

Projeto prático da disciplina de Inteligência Artificial desenvolvido com o objetivo de integrar exploração de dados, aprendizado de máquina, backend, interface web e inteligência artificial generativa.

A solução utiliza o dataset **Netflix Movies and TV Shows**, disponível no Kaggle, para construir um agente especialista capaz de responder perguntas sobre a base de dados, explicar o projeto, consultar títulos presentes no dataset e realizar predições classificando um conteúdo como **Movie** ou **TV Show**.

## Integrantes

- João Pedro Guzzi Guerreiro
- Eduardo Hermes Becker

## Objetivo do projeto

Construir um Agente Preditivo Especialista utilizando o dataset Netflix Movies and TV Shows.

O sistema permite:

- analisar informações do dataset;
- consultar títulos presentes na base;
- responder perguntas sobre colunas, registros, métricas e modelos;
- recusar perguntas fora do escopo do projeto;
- realizar predições com Machine Learning;
- explicar os resultados usando a API do Gemini.

## Dataset

Dataset utilizado:

**Netflix Movies and TV Shows**

Fonte: Kaggle

A base contém informações sobre títulos disponíveis na Netflix, incluindo:

- identificador;
- tipo do conteúdo;
- título;
- diretor;
- elenco;
- país;
- data de adição à plataforma;
- ano de lançamento;
- classificação indicativa;
- duração;
- categorias;
- descrição.

A variável alvo do projeto é:

```text
type
```

Classes:

```text
Movie = 0
TV Show = 1
```

## Variáveis utilizadas no modelo

As variáveis utilizadas para treinar o modelo foram:

```text
release_year
year_added
month_added
num_genres
added_delay
rating
country_first
```

As variáveis `duration`, `duration_num` e `listed_in` não foram usadas no treinamento principal por risco de vazamento de informação, pois podem indicar diretamente se o conteúdo é filme ou série.

## Modelos avaliados

Foram treinados e comparados quatro modelos:

1. Regressão Linear Múltipla adaptada para classificação
2. KNN
3. MLP
4. Naive Bayes

Métricas utilizadas:

- acurácia;
- sensibilidade;
- especificidade;
- precisão;
- F1-score;
- matriz de confusão.

## Melhor modelo

O melhor modelo foi o **MLP**.

Métricas obtidas no conjunto de teste:

```text
Acurácia: 0.7920
Sensibilidade: 0.6278
Especificidade: 0.8637
Precisão: 0.6677
F1-score: 0.6471
```

O modelo foi salvo em:

```text
artifacts/best_model.joblib
```

## Funcionalidades do agente

O chat especialista consegue responder perguntas como:

```text
Quantos registros existem no dataset?
Quais são as colunas da base?
Qual é a variável alvo?
Qual modelo teve melhor desempenho?
Por que duration não foi usada no treinamento?
Dentro do dataset existe o filme homem-aranha 3?
Qual o ano de lançamento do filme spider-man 3?
Qual a classificação indicativa do filme spider-man 3?
```

Também consegue realizar predições pelo chat:

```text
Faça uma predição com release_year 2019, year_added 2021, month_added 7, num_genres 2, added_delay 2, rating TV-MA, country_first United States
```

Perguntas fora do escopo são recusadas, por exemplo:

```text
Qual é a capital da França?
```

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- FastAPI
- Streamlit
- Gemini API
- Pydantic
- Requests
- Python Dotenv

## Estrutura do projeto

```text
Trab_IA_Final/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── best_model.joblib
│   ├── dataset_description.txt
│   ├── metrics.json
│   └── model_info.json
├── data/
│   ├── raw/
│   │   └── netflix_titles.csv
│   └── processed/
│       └── netflix_modeling.csv
├── reports/
│   └── figures/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   ├── dataset_context.py
│   │   ├── dataset_query.py
│   │   └── load_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── llm/
│   │   ├── dataset_chat_agent.py
│   │   └── gemini_agent.py
│   ├── models/
│   │   ├── chat_prediction.py
│   │   ├── evaluate_models.py
│   │   ├── predict_model.py
│   │   └── train_models.py
│   └── visualization/
│       └── generate_figures.py
├── .env.example
├── .gitignore
├── diario_de_bordo.md
├── README.md
└── requirements.txt
```

## Como executar o projeto

### 1. Clonar o repositório

```powershell
git clone <link-do-repositorio>
cd Trab_IA_Final
```

### 2. Criar ambiente virtual

```powershell
py -3.12 -m venv venv
```

### 3. Ativar ambiente virtual

```powershell
.\venv\Scripts\Activate.ps1
```

Caso ocorra erro de permissão no PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 4. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 5. Configurar chave da Gemini API

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
GEMINI_API_KEY=sua_chave_aqui
```

O arquivo `.env` não deve ser enviado para o GitHub.

### 6. Conferir se o dataset está na pasta correta

O arquivo CSV deve estar em:

```text
data/raw/netflix_titles.csv
```

### 7. Gerar base processada

```powershell
python -m src.features.build_features
```

### 8. Treinar os modelos

```powershell
python -m src.models.train_models
```

### 9. Gerar gráficos

```powershell
python -m src.visualization.generate_figures
```

### 10. Rodar backend FastAPI

Em um terminal:

```powershell
uvicorn src.api.main:app --reload
```

A documentação da API ficará disponível em:

```text
http://127.0.0.1:8000/docs
```

### 11. Rodar interface Streamlit

Em outro terminal:

```powershell
streamlit run app/streamlit_app.py
```

A interface ficará disponível em:

```text
http://localhost:8501
```

## Endpoints da API

### GET `/health`

Verifica se a API está em execução.

### GET `/model-info`

Retorna informações sobre o modelo treinado.

### POST `/predict`

Realiza uma predição usando o modelo treinado.

Exemplo de entrada:

```json
{
  "release_year": 2019,
  "year_added": 2021,
  "month_added": 7,
  "num_genres": 2,
  "added_delay": 2,
  "rating": "TV-MA",
  "country_first": "United States"
}
```

### POST `/chat`

Envia uma pergunta para o agente especialista.

Exemplo:

```json
{
  "message": "Quantos registros existem no dataset?"
}
```

## Gráficos gerados

Os gráficos ficam em:

```text
reports/figures/
```

Figuras geradas:

- frequência por tipo de conteúdo;
- frequência por classificação indicativa;
- top 10 países principais;
- matriz de correlação;
- box plot do ano de lançamento por tipo;
- box plot do atraso de adição por tipo;
- comparação das métricas dos modelos;
- matrizes de confusão dos modelos.

## Diário de Bordo de Contribuições

### João Pedro Guzzi Guerreiro

- Configuração do ambiente Python.
- Criação do ambiente virtual.
- Instalação das dependências.
- Organização inicial da estrutura do projeto.
- Implementação da leitura do dataset.
- Implementação do pré-processamento.
- Treinamento dos modelos.
- Implementação da API com FastAPI.
- Integração com Gemini.
- Implementação do agente especialista do dataset.
- Implementação de busca de títulos no CSV.
- Implementação de predição pelo chat.
- Implementação da interface Streamlit.
- Geração dos gráficos exploratórios e gráficos de métricas.

### Eduardo Hermes Becker

- Apoio na definição do problema.
- Apoio na análise dos modelos.
- Apoio na validação dos testes.
- Apoio na revisão da apresentação e relatório.

## Observações

O arquivo `.env` não deve ser versionado, pois contém a chave de API.

O arquivo `best_model.joblib` pode ser gerado novamente executando o treinamento dos modelos.

O agente especialista foi projetado para responder apenas perguntas relacionadas ao dataset, ao projeto, aos modelos, às métricas, ao pré-processamento e ao funcionamento da solução.
