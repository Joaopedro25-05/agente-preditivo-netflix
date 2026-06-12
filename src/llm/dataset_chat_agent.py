import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.data.dataset_context import build_dataset_context_text


OUT_OF_SCOPE_MESSAGE = (
    "Não posso responder essa pergunta, pois ela está fora do escopo deste agente. "
    "Este chat responde apenas sobre o dataset Netflix Movies and TV Shows, "
    "o projeto de classificação Movie/TV Show, os modelos treinados, métricas, "
    "pré-processamento, variáveis utilizadas e funcionamento da solução."
)


def is_question_related_to_project(question: str) -> bool:
    """
    Verifica de forma simples se a pergunta parece estar relacionada ao escopo do projeto.

    Essa checagem é uma primeira barreira para evitar perguntas fora do tema.
    O Gemini também recebe instruções para recusar perguntas fora do escopo.
    """
    question_lower = question.lower()

    allowed_terms = [
        "dataset",
        "base",
        "dados",
        "netflix",
        "movie",
        "filme",
        "filmes",
        "tv show",
        "série",
        "series",
        "séries",
        "classificação",
        "classificar",
        "predição",
        "previsão",
        "modelo",
        "mlp",
        "knn",
        "naive bayes",
        "regressão",
        "acurácia",
        "sensibilidade",
        "especificidade",
        "precisão",
        "f1",
        "métrica",
        "métricas",
        "coluna",
        "colunas",
        "variável",
        "variáveis",
        "target",
        "type",
        "rating",
        "duration",
        "listed_in",
        "country",
        "release_year",
        "date_added",
        "year_added",
        "month_added",
        "added_delay",
        "num_genres",
        "country_first",
        "kaggle",
        "gráfico",
        "correlação",
        "boxplot",
        "box plot",
        "frequência",
        "pré-processamento",
        "treinamento",
        "teste",
        "matriz de confusão",
        "vazamento",
        "leakage",
    ]

    return any(term in question_lower for term in allowed_terms)


def build_chat_system_instruction() -> str:
    """
    Monta o System Prompt do agente especialista.

    O agente deve responder apenas sobre o dataset e o projeto.
    """
    dataset_context = build_dataset_context_text()

    return f"""
Você é um agente especialista exclusivamente no dataset Netflix Movies and TV Shows e no projeto de Machine Learning desenvolvido sobre ele.

Seu escopo permitido:
- explicar o dataset Netflix Movies and TV Shows;
- explicar as colunas da base;
- explicar a variável alvo type;
- explicar a classificação Movie ou TV Show;
- explicar pré-processamento e engenharia de atributos;
- explicar valores ausentes, distribuição das classes e características da base;
- explicar os modelos usados: Regressão Linear Múltipla adaptada, KNN, MLP e Naive Bayes;
- explicar métricas: acurácia, sensibilidade, especificidade, precisão, F1-score e matriz de confusão;
- explicar por que algumas variáveis não foram usadas no treino;
- explicar o funcionamento geral do agente preditivo;
- orientar o usuário sobre quais campos informar para realizar uma predição.

Regras obrigatórias:
1. Responda apenas perguntas relacionadas ao dataset, ao modelo, às métricas, ao pré-processamento ou à arquitetura deste projeto.
2. Se a pergunta estiver fora do escopo, recuse educadamente e diga que só pode responder sobre o dataset e o projeto.
3. Não invente informações que não estejam no contexto fornecido.
4. Não diga que conhece títulos reais da Netflix além dos dados estruturais do dataset.
5. Não faça afirmações absolutas sobre predições; trate resultados como estimativas estatísticas.
6. Quando houver limitação ou incerteza, explique isso claramente.
7. Responda em português.
8. Use linguagem clara e objetiva.
9. Não responda perguntas gerais de programação, notícias, esportes, religião, política, saúde, receitas, entretenimento ou qualquer outro tema externo ao projeto.
10. Se o usuário pedir uma predição, explique que ele deve informar estes campos:
   release_year, year_added, month_added, num_genres, added_delay, rating e country_first.

Contexto confiável do projeto:
{dataset_context}
""".strip()


def build_chat_prompt(user_question: str) -> str:
    """
    Monta a pergunta do usuário para o agente.
    """
    return f"""
Pergunta do usuário:
{user_question}

Responda respeitando estritamente o escopo do agente especialista.
""".strip()


def generate_dataset_chat_response(user_question: str) -> str:
    """
    Gera resposta do agente especialista.

    A função aplica:
    - filtro local de escopo;
    - system prompt restritivo;
    - fallback caso a API falhe.
    """
    if not is_question_related_to_project(user_question):
        return OUT_OF_SCOPE_MESSAGE

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "A pergunta está dentro do escopo do projeto, mas a chave da Gemini API "
            "não foi encontrada no arquivo .env."
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_chat_prompt(user_question),
            config=types.GenerateContentConfig(
                system_instruction=build_chat_system_instruction(),
                temperature=0.2,
            ),
        )

        if not response.text:
            return (
                "Não foi possível gerar uma resposta no momento. "
                "Tente reformular a pergunta dentro do escopo do dataset."
            )

        return response.text.strip()

    except Exception:
        return (
            "A pergunta está dentro do escopo do projeto, mas não foi possível consultar "
            "o agente Gemini neste momento. Tente novamente em instantes."
        )


if __name__ == "__main__":
    test_questions: list[dict[str, Any]] = [
        {
            "description": "Pergunta dentro do escopo",
            "question": "Quantos registros existem no dataset e qual é a variável alvo?",
        },
        {
            "description": "Pergunta fora do escopo",
            "question": "Qual é a capital da França?",
        },
    ]

    for item in test_questions:
        print("=" * 80)
        print(item["description"])
        print("Pergunta:", item["question"])
        print("Resposta:")
        print(generate_dataset_chat_response(item["question"]))
        print()