import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BEST_MODEL_PATH = Path("artifacts/best_model.joblib")
MODEL_INFO_PATH = Path("artifacts/model_info.json")


def load_model() -> Any:
    """
    Carrega o melhor modelo treinado.
    """
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {BEST_MODEL_PATH}. "
            "Execute primeiro: python -m src.models.train_models"
        )

    return joblib.load(BEST_MODEL_PATH)


def load_model_info() -> dict:
    """
    Carrega as informações do modelo treinado.
    """
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de informações não encontrado em: {MODEL_INFO_PATH}."
        )

    with MODEL_INFO_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_prediction_label(prediction_code: int) -> str:
    """
    Converte o código da predição em texto.
    """
    if prediction_code == 0:
        return "Movie"

    if prediction_code == 1:
        return "TV Show"

    return "Classe desconhecida"


def predict_single(input_data: dict) -> dict:
    """
    Realiza uma predição individual.

    Args:
        input_data: Dicionário com as variáveis esperadas pelo modelo.

    Returns:
        Dicionário com a predição, probabilidade e informações do modelo.
    """
    model = load_model()
    model_info = load_model_info()

    feature_columns = model_info["feature_columns"]

    input_df = pd.DataFrame([input_data])

    missing_columns = [
        column for column in feature_columns if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Campos ausentes na entrada: "
            + ", ".join(missing_columns)
        )

    input_df = input_df[feature_columns]

    prediction_code = int(model.predict(input_df)[0])
    prediction_label = get_prediction_label(prediction_code)

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        probability = round(float(probabilities[prediction_code]), 4)

    return {
        "prediction_code": prediction_code,
        "prediction_label": prediction_label,
        "probability": probability,
        "model_used": model_info["best_model_name"],
        "input_data": input_data,
    }


if __name__ == "__main__":
    example_input = {
        "release_year": 2019,
        "year_added": 2021,
        "month_added": 7,
        "num_genres": 2,
        "added_delay": 2,
        "rating": "TV-MA",
        "country_first": "United States",
    }

    result = predict_single(example_input)

    print("Predição realizada com sucesso.")
    print(f"Modelo utilizado: {result['model_used']}")
    print(f"Resultado: {result['prediction_label']}")
    print(f"Código da classe: {result['prediction_code']}")
    print(f"Probabilidade: {result['probability']}")