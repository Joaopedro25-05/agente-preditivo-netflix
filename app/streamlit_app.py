import requests
import streamlit as st


API_CHAT_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(
    page_title="Chat Especialista Netflix Dataset",
    page_icon="🎬",
    layout="centered",
)


st.title("Chat Especialista — Netflix Movies and TV Shows")

st.write(
    """
    Este chat responde apenas sobre o dataset Netflix Movies and TV Shows,
    o projeto de classificação Movie/TV Show, os modelos treinados,
    métricas, pré-processamento e funcionamento da solução.
    """
)

st.warning(
    "Perguntas fora do escopo do dataset ou do projeto serão recusadas pelo agente."
)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Sou o agente especialista no dataset Netflix Movies and TV Shows. "
                "Você pode perguntar sobre a base, colunas, métricas, modelos, "
                "pré-processamento ou funcionamento do projeto."
            ),
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_message = st.chat_input("Digite sua pergunta sobre o dataset ou o projeto...")


if user_message:
    history_for_api = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages[-10:]
    ]

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.write(user_message)

    try:
        with st.spinner("Consultando agente especialista..."):
            response = requests.post(
                API_CHAT_URL,
                json={
                    "message": user_message,
                    "history": history_for_api,
                },
                timeout=60,
            )

        if response.status_code == 200:
            assistant_response = response.json().get(
                "response",
                "A API não retornou uma resposta válida.",
            )
        else:
            assistant_response = (
                f"A API retornou um erro: {response.status_code} - {response.text}"
            )

    except requests.exceptions.ConnectionError:
        assistant_response = (
            "Não foi possível conectar à API. "
            "Verifique se o FastAPI está rodando em http://127.0.0.1:8000."
        )

    except requests.exceptions.Timeout:
        assistant_response = "A requisição demorou demais para responder."

    except Exception as error:
        assistant_response = f"Erro inesperado: {error}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )

    with st.chat_message("assistant"):
        st.write(assistant_response)