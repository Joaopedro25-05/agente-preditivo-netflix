from typing import Any
from unittest import result

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm.gemini_agent import generate_prediction_explanation
from src.models.predict_model import load_model_info, predict_single
from src.llm.dataset_chat_agent import generate_dataset_chat_response


app = FastAPI(
    title="Agente Preditivo Netflix",
    description="API para classificar títulos da Netflix como Movie ou TV Show.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """
    Dados de entrada esperados pelo modelo preditivo.
    """

    release_year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Ano de lançamento do título.",
    )
    year_added: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Ano em que o título foi adicionado à Netflix.",
    )
    month_added: int = Field(
        ...,
        ge=1,
        le=12,
        description="Mês em que o título foi adicionado à Netflix.",
    )
    num_genres: int = Field(
        ...,
        ge=1,
        le=20,
        description="Quantidade de gêneros/categorias associados ao título.",
    )
    added_delay: int = Field(
        ...,
        ge=-100,
        le=150,
        description="Diferença entre o ano de adição e o ano de lançamento.",
    )
    rating: str = Field(
        ...,
        min_length=1,
        description="Classificação indicativa do título, por exemplo TV-MA, TV-14 ou PG.",
    )
    country_first: str = Field(
        ...,
        min_length=1,
        description="Primeiro país listado no campo country.",
    )


class PredictionResponse(BaseModel):
    """
    Resposta retornada pela API após a predição.
    """

    prediction_code: int
    prediction_label: str
    probability: float | None
    model_used: str
    input_data: dict[str, Any]
    explanation: str

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    """
    Resposta retornada pelo agente especialista.
    """

    response: str


@app.get("/")
def root() -> dict:
    """
    Rota inicial da API.
    """
    return {
        "message": "API do Agente Preditivo Netflix em execução.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict:
    """
    Verifica se a API está online.
    """
    return {
        "status": "ok",
        "service": "agente-preditivo-netflix",
    }


@app.get("/model-info")
def get_model_info() -> dict:
    """
    Retorna informações sobre o modelo treinado.
    """
    try:
        return load_model_info()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    """
    Recebe os dados de entrada, executa o modelo treinado e retorna a predição.
    """
    try:
        input_data = request.model_dump()
        result = predict_single(input_data)
        result["explanation"] = generate_prediction_explanation(result)
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado ao realizar a predição: {error}",
        ) from error
    
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        history = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.history
        ]

        response = generate_dataset_chat_response(
            user_question=request.message,
            history=history,
        )

        return ChatResponse(response=response)

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))