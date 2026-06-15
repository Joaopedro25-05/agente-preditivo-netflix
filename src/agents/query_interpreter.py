import json
import os
import re
import unicodedata
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path


ALLOWED_ACTIONS = [
    "describe_dataset",
    "filter_records",
    "get_title_info",
    "count_records",
    "model_info",
    "compare_models",
    "predict",
    "unsupported_field",
    "out_of_scope",
]


DEFAULT_INTERPRETATION = {
    "action": "out_of_scope",
    "filters": {},
    "fields": [],
    "limit": 10,
    "prediction_input": {},
    "explanation": "Não foi possível interpretar a pergunta dentro do escopo do projeto.",
}

CACHE_PATH = Path("artifacts/query_interpretation_cache.json")


def load_interpretation_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}

    try:
        with CACHE_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_interpretation_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def build_cache_key(user_question: str, history: list[dict] | None = None) -> str:
    history_text = ""

    if history:
        recent_history = history[-4:]
        history_text = json.dumps(recent_history, ensure_ascii=False, sort_keys=True)

    return normalize_question_text(user_question) + "|" + normalize_question_text(history_text)

def normalize_question_text(text: str) -> str:
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_unsupported_field_question(user_question: str) -> bool:
    """
    Detecta perguntas sobre campos que não existem no dataset.

    Exemplo: nota IMDb, avaliação dos usuários, filmes mais bem avaliados.
    """
    normalized_question = normalize_question_text(user_question)

    unsupported_terms = [
        "nota",
        "notas",
        "avaliacao",
        "avaliacoes",
        "avaliado",
        "avaliados",
        "avaliada",
        "avaliadas",
        "mais bem avaliados",
        "melhor avaliados",
        "melhores avaliados",
        "imdb",
        "rotten tomatoes",
        "metacritic",
        "score",
        "pontuacao",
        "estrelas",
        "review",
        "reviews",
    ]

    return any(term in normalized_question for term in unsupported_terms)


def build_unsupported_field_interpretation(user_question: str) -> dict[str, Any]:
    """
    Retorna interpretação local para perguntas sobre campos inexistentes.
    """
    return {
        "action": "unsupported_field",
        "filters": {
            "type": None,
            "title_contains": None,
            "release_year": None,
            "rating": None,
            "country_contains": None,
            "listed_in_contains": None,
        },
        "fields": [],
        "limit": 10,
        "prediction_input": {
            "release_year": None,
            "year_added": None,
            "month_added": None,
            "num_genres": None,
            "added_delay": None,
            "rating": None,
            "country_first": None,
        },
        "explanation": (
            "A pergunta solicita nota de avaliação, avaliação de usuários ou ranking por nota, "
            "mas esse campo não existe no dataset."
        ),
    }


def build_interpreter_system_instruction() -> str:
    """
    Cria a instrução do interpretador.

    Esta etapa não responde ao usuário final.
    Ela apenas transforma a pergunta em JSON estruturado.
    """
    return """
Você é um interpretador de perguntas para um agente especialista no dataset Netflix Movies and TV Shows.

Sua função é transformar a pergunta do usuário em um JSON estruturado.

Você não deve responder ao usuário final.
Você deve retornar apenas JSON válido, sem markdown, sem explicações fora do JSON.

Escopo permitido:
- perguntas sobre o dataset Netflix Movies and TV Shows;
- perguntas sobre colunas, registros, títulos, categorias, anos, países, classificação indicativa, duração, elenco, diretor e descrição;
- perguntas sobre o projeto de classificação Movie/TV Show;
- perguntas sobre modelos treinados, métricas e pré-processamento;
- pedidos de predição usando os campos do modelo.

Escopo proibido:
- perguntas gerais fora do dataset ou fora do projeto;
- notícias, esportes, política, saúde, religião, programação geral, receitas, curiosidades gerais ou qualquer tema externo.

Ações disponíveis:

1. describe_dataset
Use quando o usuário perguntar sobre estrutura geral do dataset, colunas, quantidade de registros, valores ausentes ou variável alvo.

2. filter_records
Use quando o usuário pedir listagens ou filtros de registros.
Exemplos:
- liste um filme da categoria comedy
- me diga os filmes com lançamento em 2021
- mostre séries do Japão
- quais filmes PG-13 existem
- liste conteúdos de terror
- mostre 5 séries coreanas

3. get_title_info
Use quando o usuário perguntar sobre um título específico.
Exemplos:
- existe o filme Spider-Man 3?
- qual o ano de lançamento do filme Homem-Aranha 3?
- qual a classificação indicativa de Spider-Man 3?
- qual a duração de Breaking Bad?
- quem é o diretor de determinado título?

4. count_records
Use quando o usuário pedir contagem.
Exemplos:
- quantos filmes existem?
- quantas séries são do Japão?
- quantos títulos foram lançados em 2021?
- quantos registros são da categoria comedy?

5. model_info
Use quando o usuário perguntar sobre o modelo escolhido, variáveis usadas, variável alvo, pré-processamento ou funcionamento geral do modelo.

6. compare_models
Use quando o usuário perguntar sobre métricas, comparação entre Regressão Linear, KNN, MLP e Naive Bayes, melhor modelo, acurácia, sensibilidade, especificidade, precisão ou F1-score.

7. predict
Use quando o usuário pedir uma predição/classificação Movie ou TV Show e fornecer ou tentar fornecer os campos:
release_year, year_added, month_added, num_genres, added_delay, rating, country_first.

8. unsupported_field
Use quando a pergunta estiver dentro do contexto do dataset, mas pedir uma informação que não existe na base.
Exemplos:
- traga as notas de avaliação
- quais são as notas dos filmes
- qual a nota IMDb
- qual a avaliação dos usuários
- quais os filmes mais bem avaliados
- ordene por nota

Importante:
O dataset Netflix Movies and TV Shows não possui nota de avaliação, nota IMDb, Rotten Tomatoes, avaliação média ou nota de usuários.
A coluna rating não significa nota. No dataset, rating representa classificação indicativa, como TV-MA, PG-13, TV-14, TV-PG.

9. out_of_scope
Use quando a pergunta não estiver relacionada ao dataset nem ao projeto.

Colunas disponíveis no dataset original:
show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description.

Campos usados no modelo preditivo:
release_year, year_added, month_added, num_genres, added_delay, rating, country_first.

Mapeamentos importantes:
- filme, filmes, movie, movies => type = Movie
- série, séries, serie, series, tv show => type = TV Show
- comédia, comedy, comedies => listed_in_contains = Comedy
- ação, acao, action => listed_in_contains = Action
- terror, horror => listed_in_contains = Horror
- documentário, documentario => listed_in_contains = Documentary
- ficção científica, ficcao cientifica, sci-fi => listed_in_contains = Sci-Fi
- Estados Unidos, EUA => country_contains = United States
- Japão, Japao => country_contains = Japan
- Coreia do Sul => country_contains = South Korea
- Brasil => country_contains = Brazil

Formato obrigatório da resposta:

{
  "action": "uma das ações disponíveis: describe_dataset, filter_records, get_title_info, count_records, model_info, compare_models, predict, unsupported_field, out_of_scope",
  "filters": {
    "type": null,
    "title_contains": null,
    "release_year": null,
    "rating": null,
    "country_contains": null,
    "listed_in_contains": null
  },
  "fields": [],
  "limit": 10,
  "prediction_input": {
    "release_year": null,
    "year_added": null,
    "month_added": null,
    "num_genres": null,
    "added_delay": null,
    "rating": null,
    "country_first": null
  },
  "explanation": "resumo curto da intenção interpretada"
}

Regras:
- Retorne apenas JSON válido.
- Use null para campos desconhecidos.
- Em perguntas como "liste um filme", use limit = 1.
- Em perguntas como "liste 5 filmes", use limit = 5.
- Em perguntas sem quantidade explícita, use limit = 10.
- Para get_title_info, preencha title_contains com o título provável.
- Para perguntas sobre ano de lançamento de um título, inclua "release_year" em fields.
- Para perguntas sobre classificação indicativa de um título, inclua "rating" em fields.
- Para perguntas sobre país de um título, inclua "country" em fields.
- Para perguntas sobre categoria/gênero de um título, inclua "listed_in" em fields.
- Para perguntas sobre duração de um título, inclua "duration" em fields.
- Para perguntas sobre diretor de um título, inclua "director" em fields.
- Para perguntas sobre elenco de um título, inclua "cast" em fields.
- Para perguntas sobre descrição/sinopse de um título, inclua "description" em fields.
""".strip()

def build_history_context(history: list[dict] | None, max_messages: int = 8) -> str:
    """
    Monta um resumo simples do histórico recente da conversa.

    O objetivo é permitir que o interpretador entenda perguntas de continuação,
    como: "traga as notas de avaliação", "e desses, quais são séries?", etc.
    """
    if not history:
        return "Sem histórico anterior."

    recent_messages = history[-max_messages:]

    lines = []

    for message in recent_messages:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()

        if not content:
            continue

        if len(content) > 1200:
            content = content[:1200] + "..."

        lines.append(f"{role}: {content}")

    if not lines:
        return "Sem histórico anterior."

    return "\n".join(lines)


def extract_json_from_text(text: str) -> dict[str, Any]:
    """
    Extrai um objeto JSON de uma resposta textual.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return DEFAULT_INTERPRETATION.copy()

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return DEFAULT_INTERPRETATION.copy()


def normalize_interpretation(data: dict[str, Any]) -> dict[str, Any]:
    """
    Garante que o JSON interpretado tenha a estrutura esperada.
    """
    action = data.get("action")

    if action not in ALLOWED_ACTIONS:
        action = "out_of_scope"

    filters = data.get("filters") or {}
    prediction_input = data.get("prediction_input") or {}

    normalized = {
        "action": action,
        "filters": {
            "type": filters.get("type"),
            "title_contains": filters.get("title_contains"),
            "release_year": filters.get("release_year"),
            "rating": filters.get("rating"),
            "country_contains": filters.get("country_contains"),
            "listed_in_contains": filters.get("listed_in_contains"),
        },
        "fields": data.get("fields") or [],
        "limit": data.get("limit") or 10,
        "prediction_input": {
            "release_year": prediction_input.get("release_year"),
            "year_added": prediction_input.get("year_added"),
            "month_added": prediction_input.get("month_added"),
            "num_genres": prediction_input.get("num_genres"),
            "added_delay": prediction_input.get("added_delay"),
            "rating": prediction_input.get("rating"),
            "country_first": prediction_input.get("country_first"),
        },
        "explanation": data.get("explanation") or "",
    }

    return normalized


def interpret_user_question(
    user_question: str,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Usa Gemini para interpretar a pergunta do usuário como uma ação estruturada.

    O histórico é enviado para permitir perguntas de continuidade.
    """
    if is_unsupported_field_question(user_question):
        return build_unsupported_field_interpretation(user_question)

    cache = load_interpretation_cache()
    cache_key = build_cache_key(user_question, history)

    if cache_key in cache:
        return cache[cache_key]

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            **DEFAULT_INTERPRETATION,
            "explanation": "Chave GEMINI_API_KEY não encontrada.",
        }

    history_context = build_history_context(history)

    prompt = f"""
Histórico recente da conversa:
{history_context}

Pergunta atual do usuário:
{user_question}

Interprete a pergunta atual considerando o histórico.
Retorne apenas o JSON estruturado.
""".strip()

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=build_interpreter_system_instruction(),
                temperature=0.0,
            ),
        )

        if not response.text:
            return DEFAULT_INTERPRETATION.copy()

        parsed = extract_json_from_text(response.text)

        normalized = normalize_interpretation(parsed)

        if normalized.get("action") != "out_of_scope":
            cache[cache_key] = normalized
            save_interpretation_cache(cache)

        return normalized

    except Exception as error:
        return {
            **DEFAULT_INTERPRETATION,
            "explanation": f"Erro ao interpretar pergunta: {error}",
        }


if __name__ == "__main__":
    test_history = [
        {
            "role": "user",
            "content": "quais os 5 primeiros filmes mais bem avaliados em 2020?",
        },
        {
            "role": "assistant",
            "content": (
                "Encontrei 517 registro(s) no dataset. "
                "Exibindo 5 resultados: Dick Johnson Is Dead, "
                "Europe's Most Dangerous Man: Otto Skorzeny in Spain, "
                "Tughlaq Durbar, Omo Ghetto: the Saga, Shadow Parties."
            ),
        },
    ]

    test_questions = [
        "traga as notas de avaliação",
        "liste um filme da categoria comedy",
        "Qual é a capital da França?",
    ]

    for question in test_questions:
        print("=" * 80)
        print(question)
        print(
            json.dumps(
                interpret_user_question(
                    user_question=question,
                    history=test_history,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        print()