import re
import unicodedata

import pandas as pd

from src.data.load_data import load_netflix_dataset


def normalize_text(text: object) -> str:
    """
    Normaliza texto para facilitar buscas:
    - remove acentos;
    - deixa minúsculo;
    - remove caracteres especiais;
    - reduz espaços duplicados.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_search_terms(search_term: str) -> list[str]:
    """
    Gera variações do termo pesquisado.

    Isso ajuda quando o usuário pergunta em português,
    mas o dataset contém o título em inglês.
    """
    normalized = normalize_text(search_term)

    terms = [normalized]

    alias_replacements = {
        "homem aranha": "spider man",
        "aranhaverso": "spider verse",
    }

    for portuguese_term, english_term in alias_replacements.items():
        if portuguese_term in normalized:
            terms.append(normalized.replace(portuguese_term, english_term))

    unique_terms = []

    for term in terms:
        if term and term not in unique_terms:
            unique_terms.append(term)

    return unique_terms


def detect_requested_field(question: str) -> str | None:
    """
    Detecta se o usuário perguntou por uma coluna específica do dataset.
    """
    normalized_question = normalize_text(question)

    field_keywords = {
        "type": [
            "tipo",
            "movie ou tv show",
            "filme ou serie",
            "filme ou tv show",
            "classe",
        ],
        "release_year": [
            "ano de lancamento",
            "ano lancamento",
            "ano foi lancado",
            "quando foi lancado",
            "release year",
        ],
        "rating": [
            "classificacao indicativa",
            "classificacao",
            "indicativa",
            "rating",
        ],
        "duration": [
            "duracao",
            "tempo",
            "quantos minutos",
            "temporadas",
            "duration",
        ],
        "country": [
            "pais",
            "origem",
            "country",
        ],
        "date_added": [
            "data de adicao",
            "quando foi adicionado",
            "adicionado a netflix",
            "entrou na netflix",
            "date added",
        ],
        "listed_in": [
            "categoria",
            "categorias",
            "genero",
            "generos",
            "listed in",
            "listed_in",
        ],
        "director": [
            "diretor",
            "direcao",
            "director",
        ],
        "cast": [
            "elenco",
            "atores",
            "atrizes",
            "cast",
        ],
        "description": [
            "descricao",
            "sinopse",
            "sobre o que",
            "description",
        ],
        "title": [
            "titulo",
            "nome",
            "title",
        ],
    }

    for field, keywords in field_keywords.items():
        if any(keyword in normalized_question for keyword in keywords):
            return field

    return None


def extract_title_from_question(question: str) -> str:
    """
    Tenta extrair o possível título pesquisado a partir da pergunta do usuário.
    """
    question_clean = question.strip()
    question_lower = question_clean.lower()

    patterns = [
        r"existe\s+(?:o\s+filme\s+|a\s+série\s+|a\s+serie\s+|o\s+título\s+|o\s+titulo\s+)?(.+?)\??$",
        r"tem\s+(?:o\s+filme\s+|a\s+série\s+|a\s+serie\s+|o\s+título\s+|o\s+titulo\s+)?(.+?)\??$",
        r"(?:do|da|de)\s+(?:filme|série|serie|título|titulo)\s+(.+?)\??$",
        r"(?:o|a)\s+(?:filme|série|serie|título|titulo)\s+(.+?)\s+existe",
        r"(?:filme|série|serie|título|titulo)\s+(.+?)\??$",
        r"procure\s+(?:por\s+)?(.+?)\??$",
        r"busque\s+(?:por\s+)?(.+?)\??$",
        r"aparece\s+(?:no\s+dataset\s+)?(.+?)\??$",
    ]

    for pattern in patterns:
        match = re.search(pattern, question_lower, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?.!")

    return question_clean.strip(" ?.!")    


def search_title_in_dataset(search_term: str, max_results: int = 10) -> dict:
    """
    Busca um título no dataset.

    Primeiro tenta encontrar correspondência exata.
    Se não encontrar, busca correspondência parcial.
    Também testa variações simples de tradução.
    """
    df = load_netflix_dataset().copy()

    df["title_normalized"] = df["title"].apply(normalize_text)

    search_terms = generate_search_terms(search_term)

    selected_columns = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]

    for term in search_terms:
        exact_matches = df[df["title_normalized"] == term]

        if not exact_matches.empty:
            results = (
                exact_matches[selected_columns]
                .head(max_results)
                .fillna("Unknown")
                .to_dict(orient="records")
            )

            return {
                "search_term": search_term,
                "matched_term": term,
                "match_type": "exact",
                "total_found": int(exact_matches.shape[0]),
                "results": results,
            }

    for term in search_terms:
        partial_matches = df[
            df["title_normalized"].str.contains(
                term,
                na=False,
                regex=False,
            )
        ]

        if not partial_matches.empty:
            results = (
                partial_matches[selected_columns]
                .head(max_results)
                .fillna("Unknown")
                .to_dict(orient="records")
            )

            return {
                "search_term": search_term,
                "matched_term": term,
                "match_type": "partial",
                "total_found": int(partial_matches.shape[0]),
                "results": results,
            }

    return {
        "search_term": search_term,
        "matched_term": None,
        "match_type": "not_found",
        "total_found": 0,
        "results": [],
    }


def get_field_label(field: str) -> str:
    """
    Retorna um nome amigável para a coluna.
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


def format_single_field_response(search_result: dict, requested_field: str) -> str:
    """
    Formata resposta quando o usuário perguntou por uma coluna específica.
    """
    search_term = search_result["search_term"]
    total_found = search_result["total_found"]
    results = search_result["results"]

    if total_found == 0:
        return (
            f"Não encontrei `{search_term}` no dataset Netflix Movies and TV Shows. "
            "Isso significa apenas que esse título não aparece nesta base específica, "
            "não necessariamente que ele nunca esteve disponível na Netflix."
        )

    response = ""

    for index, item in enumerate(results, start=1):
        title = item.get("title", "Título desconhecido")
        value = item.get(requested_field, "Unknown")
        field_label = get_field_label(requested_field)

        if total_found == 1:
            response += (
                f"No dataset, o {field_label} de **{title}** é: "
                f"**{value}**."
            )
        else:
            response += (
                f"{index}. **{title}**\n"
                f"   - {field_label.capitalize()}: {value}\n\n"
            )

    if total_found > len(results):
        response += (
            f"\nForam encontrados {total_found} resultados no total, "
            f"mas exibi apenas os primeiros {len(results)}."
        )

    return response.strip()


def format_title_search_response(search_result: dict, requested_field: str | None = None) -> str:
    """
    Formata a resposta da busca de título para o usuário.
    """
    if requested_field:
        return format_single_field_response(search_result, requested_field)

    search_term = search_result["search_term"]
    matched_term = search_result["matched_term"]
    total_found = search_result["total_found"]
    results = search_result["results"]
    match_type = search_result["match_type"]

    if total_found == 0:
        return (
            f"Não encontrei `{search_term}` no dataset Netflix Movies and TV Shows. "
            "Isso significa apenas que esse título não aparece nesta base específica, "
            "não necessariamente que ele nunca esteve disponível na Netflix."
        )

    if match_type == "exact":
        if normalize_text(search_term) == matched_term:
            response = f"Sim. Encontrei `{search_term}` no dataset.\n\n"
        else:
            response = (
                f"Sim. Encontrei uma correspondência para `{search_term}` no dataset. "
                f"O título aparece na base como uma variação equivalente a `{matched_term}`.\n\n"
            )
    else:
        response = (
            f"Não encontrei uma correspondência exata para `{search_term}`, "
            f"mas encontrei {total_found} resultado(s) parecido(s) no dataset.\n\n"
        )

    for index, item in enumerate(results, start=1):
        response += (
            f"{index}. **{item['title']}**\n"
            f"   - Tipo: {item['type']}\n"
            f"   - Ano de lançamento: {item['release_year']}\n"
            f"   - Classificação indicativa: {item['rating']}\n"
            f"   - Duração: {item['duration']}\n"
            f"   - País: {item['country']}\n"
            f"   - Categorias: {item['listed_in']}\n\n"
        )

    if total_found > len(results):
        response += (
            f"Foram encontrados {total_found} resultados no total, "
            f"mas exibi apenas os primeiros {len(results)}."
        )

    return response.strip()


def is_title_search_question(question: str) -> bool:
    """
    Identifica perguntas que parecem consultar um título específico no dataset.
    """
    question_lower = question.lower()
    normalized_question = normalize_text(question)

    title_indicators = [
        "filme",
        "série",
        "serie",
        "título",
        "titulo",
        "existe",
        "tem ",
        "procure",
        "busque",
        "aparece no dataset",
        "está no dataset",
        "esta no dataset",
        "dentro do dataset",
    ]

    field_indicators = [
        "ano de lancamento",
        "classificacao",
        "rating",
        "duracao",
        "pais",
        "categorias",
        "generos",
        "diretor",
        "elenco",
        "descricao",
        "sinopse",
        "data de adicao",
    ]

    has_title_indicator = any(term in question_lower for term in title_indicators)
    has_field_indicator = any(term in normalized_question for term in field_indicators)

    return has_title_indicator or has_field_indicator


def answer_title_search_question(question: str) -> str:
    """
    Responde perguntas sobre títulos consultando diretamente o CSV.

    Exemplos:
    - Existe o filme Spider-Man 3?
    - Qual o ano de lançamento do filme Spider-Man 3?
    - Qual a classificação indicativa do filme Spider-Man 3?
    """
    requested_field = detect_requested_field(question)
    title = extract_title_from_question(question)

    search_result = search_title_in_dataset(title)

    return format_title_search_response(search_result, requested_field)


if __name__ == "__main__":
    test_questions = [
        "Dentro do dataset existe o filme homem-aranha 3?",
        "Qual o ano de lançamento do filme homem-aranha 3?",
        "Qual a classificação indicativa do filme spider-man 3?",
        "Quais categorias do filme spider-man 3?",
        "Qual a capital da França?",
    ]

    for question in test_questions:
        print("=" * 80)
        print(question)
        print(answer_title_search_question(question))
        print()