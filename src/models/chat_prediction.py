import re
from typing import Any

from src.llm.gemini_agent import generate_prediction_explanation
from src.models.predict_model import predict_single


REQUIRED_FIELDS = [
    "release_year",
    "year_added",
    "month_added",
    "num_genres",
    "added_delay",
    "rating",
    "country_first",
]


def is_prediction_request(question: str) -> bool:
    """
    Identifica se o usuário está pedindo uma predição no chat.
    """
    question_lower = question.lower()

    prediction_terms = [
        "predição",
        "predicao",
        "previsão",
        "previsao",
        "classifique",
        "classificar",
        "prever",
        "faça uma predição",
        "faca uma predicao",
        "realize uma predição",
        "realize uma predicao",
        "movie ou tv show",
    ]

    return any(term in question_lower for term in prediction_terms)


def extract_numeric_field(question: str, field_name: str) -> int | None:
    """
    Extrai campos numéricos escritos no formato:
    campo valor
    campo: valor
    campo = valor
    """
    pattern = rf"{field_name}\s*[:=]?\s*(-?\d+)"

    match = re.search(
        pattern,
        question,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(1))


def extract_text_field(question: str, field_name: str) -> str | None:
    """
    Extrai campos de texto escritos no formato:
    campo valor
    campo: valor
    campo = valor

    A captura para em vírgula, ponto e vírgula ou quebra de linha.
    """
    pattern = rf"{field_name}\s*[:=]?\s*([^,;\n]+)"

    match = re.search(
        pattern,
        question,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    value = match.group(1).strip()

    if not value:
        return None

    return value


def extract_prediction_input(question: str) -> tuple[dict[str, Any], list[str]]:
    """
    Extrai os campos necessários para realizar uma predição.
    """
    input_data = {
        "release_year": extract_numeric_field(question, "release_year"),
        "year_added": extract_numeric_field(question, "year_added"),
        "month_added": extract_numeric_field(question, "month_added"),
        "num_genres": extract_numeric_field(question, "num_genres"),
        "added_delay": extract_numeric_field(question, "added_delay"),
        "rating": extract_text_field(question, "rating"),
        "country_first": extract_text_field(question, "country_first"),
    }

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if input_data.get(field) is None
    ]

    clean_input_data = {
        field: value
        for field, value in input_data.items()
        if value is not None
    }

    return clean_input_data, missing_fields


def format_missing_fields_message(missing_fields: list[str]) -> str:
    """
    Retorna uma mensagem explicando quais campos faltaram.
    """
    fields_text = ", ".join(missing_fields)

    return (
        "Para realizar a predição, preciso que você informe todos estes campos:\n\n"
        "- release_year\n"
        "- year_added\n"
        "- month_added\n"
        "- num_genres\n"
        "- added_delay\n"
        "- rating\n"
        "- country_first\n\n"
        f"Campos ausentes na sua mensagem: {fields_text}.\n\n"
        "Exemplo de pergunta:\n"
        "`Faça uma predição com release_year 2019, year_added 2021, "
        "month_added 7, num_genres 2, added_delay 2, rating TV-MA, "
        "country_first United States`"
    )


def format_prediction_response(prediction_result: dict[str, Any]) -> str:
    """
    Formata a resposta da predição para o chat.
    """
    prediction_label = prediction_result["prediction_label"]
    probability = prediction_result["probability"]
    model_used = prediction_result["model_used"]
    explanation = prediction_result.get("explanation", "")

    probability_text = (
        f"{probability * 100:.2f}%"
        if isinstance(probability, (int, float))
        else "não informada"
    )

    return (
        f"**Resultado da predição**\n\n"
        f"- Classe prevista: **{prediction_label}**\n"
        f"- Probabilidade: **{probability_text}**\n"
        f"- Modelo utilizado: **{model_used}**\n\n"
        f"**Explicação:**\n\n"
        f"{explanation}"
    )


def answer_prediction_request(question: str) -> str:
    """
    Executa uma predição solicitada pelo usuário no chat.
    """
    input_data, missing_fields = extract_prediction_input(question)

    if missing_fields:
        return format_missing_fields_message(missing_fields)

    prediction_result = predict_single(input_data)
    prediction_result["explanation"] = generate_prediction_explanation(prediction_result)

    return format_prediction_response(prediction_result)


if __name__ == "__main__":
    test_question = (
        "Faça uma predição com release_year 2019, year_added 2021, "
        "month_added 7, num_genres 2, added_delay 2, rating TV-MA, "
        "country_first United States"
    )

    print(answer_prediction_request(test_question))