import json
from pathlib import Path

import pandas as pd

from src.data.load_data import load_netflix_dataset
from src.features.build_features import build_features


DATASET_DESCRIPTION_PATH = Path("artifacts/dataset_description.txt")
METRICS_PATH = Path("artifacts/metrics.json")
MODEL_INFO_PATH = Path("artifacts/model_info.json")


def load_text_file(path: Path) -> str:
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8")


def load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_dataset_summary() -> dict:
    """
    Gera um resumo objetivo e confiável do dataset.

    Esse resumo será usado pelo agente especialista para responder
    perguntas sobre a base, evitando respostas inventadas.
    """
    raw_df = load_netflix_dataset()
    processed_df = build_features(raw_df)

    target_distribution = raw_df["type"].value_counts().to_dict()

    missing_values = raw_df.isnull().sum().to_dict()

    top_countries = (
        raw_df["country"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .replace("", "Unknown")
        .value_counts()
        .head(10)
        .to_dict()
    )

    top_ratings = (
        raw_df["rating"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .to_dict()
    )

    release_year_stats = {
        "min": int(raw_df["release_year"].min()),
        "max": int(raw_df["release_year"].max()),
        "median": float(raw_df["release_year"].median()),
    }

    summary = {
        "dataset_name": "Netflix Movies and TV Shows",
        "source": "Kaggle",
        "rows": int(raw_df.shape[0]),
        "columns": int(raw_df.shape[1]),
        "column_names": list(raw_df.columns),
        "duplicated_rows": int(raw_df.duplicated().sum()),
        "target_column": "type",
        "target_mapping": {
            "Movie": 0,
            "TV Show": 1,
        },
        "target_distribution": target_distribution,
        "missing_values": missing_values,
        "top_10_first_countries": top_countries,
        "top_10_ratings": top_ratings,
        "release_year_stats": release_year_stats,
        "model_features": [
            "release_year",
            "year_added",
            "month_added",
            "num_genres",
            "added_delay",
            "rating",
            "country_first",
        ],
        "features_not_used_due_to_leakage_risk": [
            "duration",
            "duration_num",
            "listed_in",
        ],
        "processed_rows": int(processed_df.shape[0]),
        "processed_columns": int(processed_df.shape[1]),
    }

    return summary


def build_dataset_context_text() -> str:
    """
    Monta o contexto textual que será enviado ao agente especialista.
    """
    dataset_description = load_text_file(DATASET_DESCRIPTION_PATH)
    metrics = load_json_file(METRICS_PATH)
    model_info = load_json_file(MODEL_INFO_PATH)
    summary = get_dataset_summary()

    context = f"""
DESCRIÇÃO DO DATASET:
{dataset_description}

RESUMO ESTRUTURADO DO DATASET:
{json.dumps(summary, ensure_ascii=False, indent=2)}

INFORMAÇÕES DO MODELO ESCOLHIDO:
{json.dumps(model_info, ensure_ascii=False, indent=2)}

MÉTRICAS DOS MODELOS:
{json.dumps(metrics, ensure_ascii=False, indent=2)}
""".strip()

    return context


if __name__ == "__main__":
    context_text = build_dataset_context_text()
    print(context_text)