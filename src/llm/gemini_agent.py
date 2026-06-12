import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DATASET_DESCRIPTION_PATH = Path("artifacts/dataset_description.txt")
METRICS_PATH = Path("artifacts/metrics.json")
MODEL_INFO_PATH = Path("artifacts/model_info.json")


def load_text_file(path: Path) -> str:
    """
    Carrega um arquivo de texto.
    """
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8")


def load_json_file(path: Path) -> dict:
    """
    Carrega um arquivo JSON.
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_system_instruction() -> str:
    """
    Cria a instrução fixa do agente.

    O objetivo é limitar a resposta ao contexto do projeto,
    evitando que o agente invente informações.
    """
    dataset_description = load_text_file(DATASET_DESCRIPTION_PATH)
    model_info = load_json_file(MODEL_INFO_PATH)
    metrics = load_json_file(METRICS_PATH)

    return f"""
Você é um agente especialista em interpretação de resultados de modelos de Machine Learning.

Sua função é explicar, em português claro, o resultado de uma predição feita por um modelo treinado com o dataset Netflix Movies and TV Shows.

Regras obrigatórias:
1. Explique apenas com base nas informações fornecidas.
2. Não invente dados sobre títulos reais da Netflix.
3. Não afirme que a predição é uma certeza absoluta.
4. Deixe claro que o resultado é uma estimativa estatística.
5. Use linguagem simples, adequada para usuário final.
6. Não diga que o modelo analisou variáveis que não foram enviadas.
7. Não recomende decisões fora do contexto do projeto.
8. Se a probabilidade for baixa ou moderada, mencione que há incerteza relevante.
9. Explique brevemente quais variáveis influenciaram a leitura geral.
10. Não use markdown excessivo.

Descrição do dataset:
{dataset_description}

Informações do modelo:
{json.dumps(model_info, ensure_ascii=False, indent=2)}

Métricas dos modelos:
{json.dumps(metrics, ensure_ascii=False, indent=2)}
""".strip()


def build_user_prompt(prediction_result: dict[str, Any]) -> str:
    """
    Cria o prompt variável com o resultado da predição.
    """
    return f"""
Explique o seguinte resultado de predição para o usuário final:

Resultado bruto:
{json.dumps(prediction_result, ensure_ascii=False, indent=2)}

A explicação deve conter:
- qual foi a classe prevista;
- qual foi a probabilidade retornada;
- o que isso significa em linguagem simples;
- quais campos de entrada foram usados;
- uma ressalva sobre incerteza estatística.
""".strip()


def generate_prediction_explanation(prediction_result: dict[str, Any]) -> str:
    """
    Gera uma explicação em linguagem natural usando Gemini.

    Caso a chave não exista ou ocorra erro na chamada,
    retorna uma explicação local de contingência.
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return generate_fallback_explanation(prediction_result)

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_user_prompt(prediction_result),
            config=types.GenerateContentConfig(
                system_instruction=build_system_instruction(),
                temperature=0.2,
            ),
        )

        if not response.text:
            return generate_fallback_explanation(prediction_result)

        return response.text.strip()

    except Exception:
        return generate_fallback_explanation(prediction_result)


def generate_fallback_explanation(prediction_result: dict[str, Any]) -> str:
    """
    Explicação local caso o Gemini não esteja configurado ou falhe.

    Isso evita que a aplicação quebre durante a apresentação.
    """
    prediction_label = prediction_result.get("prediction_label", "classe desconhecida")
    probability = prediction_result.get("probability")
    model_used = prediction_result.get("model_used", "modelo treinado")
    input_data = prediction_result.get("input_data", {})

    probability_text = (
    f"{probability * 100:.2f}%"
    if isinstance(probability, (int, float))
    else "não informada"
    )

    fields = ", ".join(input_data.keys())

    return (
        f"O modelo {model_used} estimou que o registro informado pertence à classe "
        f"{prediction_label}, com probabilidade de {probability_text}. "
        f"Isso significa que, com base nos padrões aprendidos no dataset da Netflix, "
        f"os dados fornecidos se aproximam mais dessa classe. "
        f"A predição utilizou os seguintes campos: {fields}. "
        f"Esse resultado deve ser interpretado como uma estimativa estatística, "
        f"não como uma certeza absoluta."
    )

if __name__ == "__main__":
    example_result = {
        "prediction_code": 1,
        "prediction_label": "TV Show",
        "probability": 0.6194,
        "model_used": "MLP",
        "input_data": {
            "release_year": 2019,
            "year_added": 2021,
            "month_added": 7,
            "num_genres": 2,
            "added_delay": 2,
            "rating": "TV-MA",
            "country_first": "United States",
        },
    }

    explanation = generate_prediction_explanation(example_result)
    print(explanation)