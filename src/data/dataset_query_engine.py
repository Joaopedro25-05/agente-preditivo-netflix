import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_data import load_netflix_dataset
from src.features.build_features import build_features
from src.llm.gemini_agent import generate_prediction_explanation
from src.models.predict_model import predict_single


METRICS_PATH = Path("artifacts/metrics.json")
MODEL_INFO_PATH = Path("artifacts/model_info.json")


REQUIRED_PREDICTION_FIELDS = [
    "release_year",
    "year_added",
    "month_added",
    "num_genres",
    "added_delay",
    "rating",
    "country_first",
]


def normalize_text(text: object) -> str:
    """
    Normaliza textos para buscas flexíveis.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_json_file(path: Path) -> dict:
    """
    Carrega um JSON se ele existir.
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_title_search_terms(title: str) -> list[str]:
    """
    Gera variações simples para busca de títulos.

    Exemplo:
    - homem aranha 3 → spider man 3
    """
    normalized = normalize_text(title)

    terms = [normalized]

    replacements = {
        "homem aranha": "spider man",
        "aranhaverso": "spider verse",
    }

    for source, target in replacements.items():
        if source in normalized:
            terms.append(normalized.replace(source, target))

    unique_terms = []

    for term in terms:
        if term and term not in unique_terms:
            unique_terms.append(term)

    return unique_terms


def get_field_label(field: str) -> str:
    """
    Nome amigável para campos do dataset.
    """
    labels = {
        "show_id": "identificador",
        "type": "tipo",
        "title": "título",
        "director": "diretor",
        "cast": "elenco",
        "country": "país",
        "date_added": "data de adição à Netflix",
        "release_year": "ano de lançamento",
        "rating": "classificação indicativa",
        "duration": "duração",
        "listed_in": "categorias",
        "description": "descrição",
    }

    return labels.get(field, field)


def prepare_raw_dataframe() -> pd.DataFrame:
    """
    Carrega o dataset bruto e cria colunas auxiliares normalizadas.
    """
    df = load_netflix_dataset().copy()

    df["title_normalized"] = df["title"].apply(normalize_text)
    df["country_normalized"] = df["country"].fillna("").apply(normalize_text)
    df["listed_in_normalized"] = df["listed_in"].fillna("").apply(normalize_text)
    df["rating_normalized"] = df["rating"].fillna("").apply(normalize_text)

    return df


def apply_interpreted_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Aplica filtros estruturados no DataFrame.
    """
    filtered_df = df.copy()

    type_filter = filters.get("type")
    title_contains = filters.get("title_contains")
    release_year = filters.get("release_year")
    rating = filters.get("rating")
    country_contains = filters.get("country_contains")
    listed_in_contains = filters.get("listed_in_contains")

    if type_filter:
        filtered_df = filtered_df[filtered_df["type"] == type_filter]

    if title_contains:
        title_terms = get_title_search_terms(title_contains)

        title_mask = pd.Series(False, index=filtered_df.index)

        for term in title_terms:
            exact_mask = filtered_df["title_normalized"] == term
            partial_mask = filtered_df["title_normalized"].str.contains(
                term,
                regex=False,
                na=False,
            )

            title_mask = title_mask | exact_mask | partial_mask

        filtered_df = filtered_df[title_mask]

    if release_year:
        filtered_df = filtered_df[filtered_df["release_year"] == int(release_year)]

    if rating:
        filtered_df = filtered_df[
            filtered_df["rating_normalized"] == normalize_text(rating)
        ]

    if country_contains:
        filtered_df = filtered_df[
            filtered_df["country_normalized"].str.contains(
                normalize_text(country_contains),
                regex=False,
                na=False,
            )
        ]

    if listed_in_contains:
        filtered_df = filtered_df[
            filtered_df["listed_in_normalized"].str.contains(
                normalize_text(listed_in_contains),
                regex=False,
                na=False,
            )
        ]

    return filtered_df


def describe_applied_filters(filters: dict) -> str:
    """
    Descreve filtros aplicados em linguagem simples.
    """
    descriptions = []

    if filters.get("type"):
        descriptions.append(f"tipo = {filters['type']}")

    if filters.get("title_contains"):
        descriptions.append(f"título contém {filters['title_contains']}")

    if filters.get("release_year"):
        descriptions.append(f"ano de lançamento = {filters['release_year']}")

    if filters.get("rating"):
        descriptions.append(f"classificação indicativa = {filters['rating']}")

    if filters.get("country_contains"):
        descriptions.append(f"país contém {filters['country_contains']}")

    if filters.get("listed_in_contains"):
        descriptions.append(f"categoria contém {filters['listed_in_contains']}")

    if not descriptions:
        return "nenhum filtro específico"

    return ", ".join(descriptions)


def format_records_list(df: pd.DataFrame, filters: dict, limit: int) -> str:
    """
    Formata listagem de registros.
    """
    if df.empty:
        return (
            "Não encontrei registros no dataset com os filtros informados.\n\n"
            f"Filtros aplicados: {describe_applied_filters(filters)}."
        )

    selected_columns = [
        "title",
        "type",
        "release_year",
        "rating",
        "duration",
        "country",
        "listed_in",
    ]

    safe_limit = max(1, min(int(limit or 10), 20))

    records = (
        df[selected_columns]
        .head(safe_limit)
        .fillna("Unknown")
        .to_dict(orient="records")
    )

    response = (
        f"Encontrei **{len(df)} registro(s)** no dataset.\n\n"
        f"Filtros aplicados: {describe_applied_filters(filters)}.\n\n"
        f"Exibindo até {safe_limit} resultado(s):\n\n"
    )

    for index, item in enumerate(records, start=1):
        response += (
            f"{index}. **{item['title']}**\n"
            f"   - Tipo: {item['type']}\n"
            f"   - Ano de lançamento: {item['release_year']}\n"
            f"   - Classificação indicativa: {item['rating']}\n"
            f"   - Duração: {item['duration']}\n"
            f"   - País: {item['country']}\n"
            f"   - Categorias: {item['listed_in']}\n\n"
        )

    return response.strip()


def execute_filter_records(interpretation: dict) -> str:
    """
    Executa ação filter_records.
    """
    df = prepare_raw_dataframe()
    filters = interpretation.get("filters") or {}
    limit = interpretation.get("limit") or 10

    filtered_df = apply_interpreted_filters(df, filters)

    return format_records_list(filtered_df, filters, limit)


def execute_count_records(interpretation: dict) -> str:
    """
    Executa ação count_records.
    """
    df = prepare_raw_dataframe()
    filters = interpretation.get("filters") or {}

    filtered_df = apply_interpreted_filters(df, filters)

    return (
        f"Encontrei **{len(filtered_df)} registro(s)** no dataset.\n\n"
        f"Filtros aplicados: {describe_applied_filters(filters)}."
    )


def execute_get_title_info(interpretation: dict) -> str:
    """
    Executa ação get_title_info.
    """
    df = prepare_raw_dataframe()
    filters = interpretation.get("filters") or {}
    fields = interpretation.get("fields") or []
    limit = interpretation.get("limit") or 10

    title = filters.get("title_contains")

    if not title:
        return "Não consegui identificar o título solicitado na pergunta."

    filtered_df = apply_interpreted_filters(df, filters)

    if filtered_df.empty:
        return (
            f"Não encontrei `{title}` no dataset Netflix Movies and TV Shows. "
            "Isso significa apenas que esse título não aparece nesta base específica."
        )

    if not fields:
        fields = [
            "type",
            "release_year",
            "rating",
            "duration",
            "country",
            "listed_in",
        ]

    safe_limit = max(1, min(int(limit or 10), 10))

    valid_fields = [
        field for field in fields
        if field in df.columns
    ]

    if not valid_fields:
        valid_fields = [
            "type",
            "release_year",
            "rating",
            "duration",
            "country",
            "listed_in",
        ]

    results = (
        filtered_df[["title"] + valid_fields]
        .head(safe_limit)
        .fillna("Unknown")
        .to_dict(orient="records")
    )

    response = ""

    if len(filtered_df) > 1:
        response += (
            f"Encontrei **{len(filtered_df)} resultado(s)** relacionados a `{title}`.\n\n"
        )

    for index, item in enumerate(results, start=1):
        response += f"{index}. **{item['title']}**\n"

        for field in valid_fields:
            response += f"   - {get_field_label(field).capitalize()}: {item[field]}\n"

        response += "\n"

    return response.strip()


def execute_describe_dataset() -> str:
    """
    Retorna resumo geral do dataset.
    """
    raw_df = load_netflix_dataset()
    processed_df = build_features(raw_df)

    missing_values = raw_df.isnull().sum()
    target_distribution = raw_df["type"].value_counts()

    response = (
        "O dataset **Netflix Movies and TV Shows** possui:\n\n"
        f"- **{raw_df.shape[0]} registros**;\n"
        f"- **{raw_df.shape[1]} colunas originais**;\n"
        f"- **{int(raw_df.duplicated().sum())} registros duplicados**.\n\n"
        "A variável alvo do projeto é **type**, com as classes:\n\n"
    )

    for class_name, count in target_distribution.items():
        response += f"- {class_name}: {count} registros\n"

    response += "\nAs colunas originais são:\n\n"

    for column in raw_df.columns:
        response += f"- {column}\n"

    response += "\nPrincipais valores ausentes:\n\n"

    for column, count in missing_values.items():
        if count > 0:
            response += f"- {column}: {int(count)}\n"

    response += (
        "\nA base processada para modelagem possui "
        f"{processed_df.shape[0]} registros e {processed_df.shape[1]} colunas."
    )

    return response


def execute_model_info() -> str:
    """
    Retorna informações do modelo final.
    """
    model_info = load_json_file(MODEL_INFO_PATH)

    if not model_info:
        return "Não encontrei o arquivo de informações do modelo."

    response = (
        "O modelo final escolhido no projeto foi:\n\n"
        f"- **{model_info.get('best_model_name')}**\n\n"
        "A escolha foi feita com base no critério:\n\n"
        f"- {model_info.get('selection_criterion')}\n\n"
        "As variáveis usadas no modelo foram:\n\n"
    )

    for feature in model_info.get("feature_columns", []):
        response += f"- {feature}\n"

    response += (
        "\nA classe positiva adotada foi **TV Show**, "
        "e a classe negativa foi **Movie**."
    )

    return response


def execute_compare_models() -> str:
    """
    Retorna comparação dos modelos.
    """
    metrics = load_json_file(METRICS_PATH)
    model_info = load_json_file(MODEL_INFO_PATH)

    if not metrics:
        return "Não encontrei o arquivo de métricas dos modelos."

    response = "Comparação dos modelos no conjunto de teste:\n\n"

    for model_name, result in metrics.items():
        test_metrics = result.get("test_metrics", {})

        response += (
            f"**{model_name}**\n"
            f"- Acurácia: {test_metrics.get('accuracy')}\n"
            f"- Sensibilidade: {test_metrics.get('sensitivity')}\n"
            f"- Especificidade: {test_metrics.get('specificity')}\n"
            f"- Precisão: {test_metrics.get('precision')}\n"
            f"- F1-score: {test_metrics.get('f1_score')}\n\n"
        )

    best_model = model_info.get("best_model_name", "não identificado")

    response += (
        f"O melhor modelo escolhido foi **{best_model}**, "
        "considerando o equilíbrio geral das métricas e o F1-score."
    )

    return response


def execute_predict(interpretation: dict) -> str:
    """
    Executa predição com o modelo treinado.
    """
    prediction_input = interpretation.get("prediction_input") or {}

    missing_fields = [
        field for field in REQUIRED_PREDICTION_FIELDS
        if prediction_input.get(field) is None
    ]

    if missing_fields:
        return (
            "Para realizar a predição, preciso que você informe todos estes campos:\n\n"
            "- release_year\n"
            "- year_added\n"
            "- month_added\n"
            "- num_genres\n"
            "- added_delay\n"
            "- rating\n"
            "- country_first\n\n"
            f"Campos ausentes: {', '.join(missing_fields)}."
        )

    input_data = {
        "release_year": int(prediction_input["release_year"]),
        "year_added": int(prediction_input["year_added"]),
        "month_added": int(prediction_input["month_added"]),
        "num_genres": int(prediction_input["num_genres"]),
        "added_delay": int(prediction_input["added_delay"]),
        "rating": str(prediction_input["rating"]),
        "country_first": str(prediction_input["country_first"]),
    }

    prediction_result = predict_single(input_data)
    prediction_result["explanation"] = generate_prediction_explanation(prediction_result)

    probability = prediction_result.get("probability")

    probability_text = (
        f"{probability * 100:.2f}%"
        if isinstance(probability, (int, float))
        else "não informada"
    )

    return (
        "**Resultado da predição**\n\n"
        f"- Classe prevista: **{prediction_result['prediction_label']}**\n"
        f"- Probabilidade: **{probability_text}**\n"
        f"- Modelo utilizado: **{prediction_result['model_used']}**\n\n"
        "**Explicação:**\n\n"
        f"{prediction_result['explanation']}"
    )

def execute_unsupported_field() -> str:
    """
    Responde perguntas sobre informações que não existem no dataset.
    """
    return (
        "O dataset **Netflix Movies and TV Shows** não possui notas de avaliação dos filmes ou séries.\n\n"
        "Ele não contém campos como nota IMDb, Rotten Tomatoes, Metacritic, avaliação média, "
        "estrelas ou avaliação dos usuários.\n\n"
        "A coluna **rating** existe no dataset, mas ela não representa nota. "
        "Nesse dataset, **rating** significa **classificação indicativa**, como TV-MA, PG-13, TV-14 ou TV-PG.\n\n"
        "Portanto, não é possível listar os conteúdos mais bem avaliados usando essa base. "
        "Posso consultar outras informações disponíveis, como título, ano de lançamento, país, duração, "
        "classificação indicativa e categorias da coluna **listed_in**."
    )

def execute_interpreted_query(interpretation: dict[str, Any]) -> str:
    """
    Executa a ação interpretada.
    """
    action = interpretation.get("action")

    if action == "describe_dataset":
        return execute_describe_dataset()

    if action == "filter_records":
        return execute_filter_records(interpretation)

    if action == "get_title_info":
        return execute_get_title_info(interpretation)

    if action == "count_records":
        return execute_count_records(interpretation)

    if action == "model_info":
        return execute_model_info()

    if action == "compare_models":
        return execute_compare_models()

    if action == "predict":
        return execute_predict(interpretation)
    
    if action == "unsupported_field":
        return execute_unsupported_field()

    return (
        "Não posso responder essa pergunta, pois ela está fora do escopo deste agente. "
        "Este chat responde apenas sobre o dataset Netflix Movies and TV Shows, "
        "o projeto de classificação Movie/TV Show, os modelos treinados, métricas, "
        "pré-processamento, variáveis utilizadas e funcionamento da solução."
    )


if __name__ == "__main__":
    test_interpretations = [
        {
            "action": "filter_records",
            "filters": {
                "type": "Movie",
                "title_contains": None,
                "release_year": None,
                "rating": None,
                "country_contains": None,
                "listed_in_contains": "Comedy",
            },
            "fields": [],
            "limit": 1,
            "prediction_input": {},
            "explanation": "Listar um filme de comédia.",
        },
        {
            "action": "filter_records",
            "filters": {
                "type": "Movie",
                "title_contains": None,
                "release_year": 2021,
                "rating": None,
                "country_contains": None,
                "listed_in_contains": None,
            },
            "fields": [],
            "limit": 10,
            "prediction_input": {},
            "explanation": "Listar filmes de 2021.",
        },
        {
            "action": "get_title_info",
            "filters": {
                "type": "Movie",
                "title_contains": "Homem-Aranha 3",
                "release_year": None,
                "rating": None,
                "country_contains": None,
                "listed_in_contains": None,
            },
            "fields": ["release_year"],
            "limit": 1,
            "prediction_input": {},
            "explanation": "Consultar ano de lançamento.",
        },
    ]

    for interpretation in test_interpretations:
        print("=" * 80)
        print(json.dumps(interpretation, ensure_ascii=False, indent=2))
        print(execute_interpreted_query(interpretation))
        print()