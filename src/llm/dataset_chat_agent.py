from src.agents.dataset_agent import run_dataset_agent


def generate_dataset_chat_response(
    user_question: str,
    history: list[dict] | None = None,
) -> str:
    """
    Ponto de entrada do chat especialista.

    Esta função é chamada pela API FastAPI.
    Ela delega a pergunta para o agente refatorado, enviando também
    o histórico recente da conversa.
    """
    return run_dataset_agent(
        user_question=user_question,
        history=history or [],
    )


if __name__ == "__main__":
    test_questions = [
        "liste um filme da categoria comedy",
        "Me diga os filmes com lançamento em 2021",
        "Qual o ano de lançamento do filme homem-aranha 3?",
        "Qual é a capital da França?",
    ]

    for question in test_questions:
        print("=" * 80)
        print("Pergunta:", question)
        print("Resposta:")
        print(generate_dataset_chat_response(question))
        print()