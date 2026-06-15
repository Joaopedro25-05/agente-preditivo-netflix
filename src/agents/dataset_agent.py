import json
from typing import Any

from src.agents.query_interpreter import interpret_user_question
from src.data.dataset_query_engine import execute_interpreted_query


def has_interpretation_error(interpretation: dict[str, Any]) -> bool:
    """
    Verifica se houve erro técnico ao interpretar a pergunta.
    """
    explanation = str(interpretation.get("explanation", "")).lower()

    error_terms = [
        "erro ao interpretar",
        "resource_exhausted",
        "quota",
        "unavailable",
        "429",
        "503",
        "gemini_api_key não encontrada",
    ]

    return any(term in explanation for term in error_terms)


def run_dataset_agent(
    user_question: str,
    history: list[dict] | None = None,
) -> str:
    """
    Executa o fluxo completo do agente especialista.

    Fluxo:
    1. Recebe pergunta em linguagem natural.
    2. Usa Gemini para interpretar a intenção considerando o histórico.
    3. Transforma a pergunta em uma ação estruturada.
    4. Executa a ação no dataset/modelo.
    5. Retorna resposta ao usuário.
    """
    interpretation = interpret_user_question(
        user_question=user_question,
        history=history or [],
    )

    if has_interpretation_error(interpretation):
        return (
            "Não foi possível interpretar sua pergunta neste momento. "
            "Isso pode ter ocorrido por limite temporário da Gemini API ou instabilidade na consulta. "
            "Tente novamente em alguns instantes."
        )

    return execute_interpreted_query(interpretation)


def debug_dataset_agent(user_question: str) -> None:
    """
    Função auxiliar para depuração.

    Mostra:
    - pergunta original;
    - JSON interpretado;
    - resposta final.
    """
    interpretation = interpret_user_question(user_question=user_question, history=[])
    response = execute_interpreted_query(interpretation)

    print("=" * 80)
    print("Pergunta:")
    print(user_question)

    print("\nInterpretação:")
    print(json.dumps(interpretation, ensure_ascii=False, indent=2))

    print("\nResposta final:")
    print(response)


if __name__ == "__main__":
    test_questions = [
        "liste um filme da categoria comedy",
        "Me diga os filmes com lançamento em 2021",
        "Qual o ano de lançamento do filme homem-aranha 3?",
        "Qual é a capital da França?",
    ]

    for question in test_questions:
        debug_dataset_agent(question)
        print()