import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_netflix_dataset
from src.features.build_features import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    build_features,
    save_processed_dataset,
)
from src.models.evaluate_models import calculate_classification_metrics, print_metrics


warnings.filterwarnings("ignore", category=ConvergenceWarning)


ARTIFACTS_DIR = Path("artifacts")
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model_info.json"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "release_year",
    "year_added",
    "month_added",
    "num_genres",
    "added_delay",
]

CATEGORICAL_FEATURES = [
    "rating",
    "country_first",
]


class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Adapta a Regressão Linear Múltipla para classificação binária.

    A regressão linear gera uma saída contínua.
    Depois aplicamos um limiar:

    - valor menor que o threshold: Movie = 0
    - valor maior ou igual ao threshold: TV Show = 1
    """

    def __init__(self, threshold: float = 0.5, fit_intercept: bool = True):
        self.threshold = threshold
        self.fit_intercept = fit_intercept

    def fit(self, x: Any, y: Any):
        self.model_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.model_.fit(x, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, x: Any):
        predictions = self.model_.predict(x)
        return (predictions >= self.threshold).astype(int)

    def predict_proba(self, x: Any):
        predictions = self.model_.predict(x)
        probabilities_positive = np.clip(predictions, 0, 1)
        probabilities_negative = 1 - probabilities_positive

        return np.column_stack(
            [
                probabilities_negative,
                probabilities_positive,
            ]
        )


def load_or_create_processed_dataset() -> pd.DataFrame:
    """
    Carrega a base processada.

    Caso ela ainda não exista, cria a base a partir do CSV bruto.
    """
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)

    raw_df = load_netflix_dataset()
    processed_df = build_features(raw_df)
    save_processed_dataset(processed_df)

    return processed_df


def build_preprocessor() -> ColumnTransformer:
    """
    Cria o pré-processador das variáveis.

    Numéricas:
    - imputação pela mediana
    - padronização

    Categóricas:
    - imputação pela moda
    - one-hot encoding
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_model_pipeline(classifier: Any) -> Pipeline:
    """
    Monta um pipeline completo com pré-processamento e classificador.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def get_models_and_parameters() -> dict:
    """
    Define os modelos e seus espaços de busca de hiperparâmetros.
    """
    return {
        "Regressao_Linear_Multipla": {
            "pipeline": build_model_pipeline(LinearRegressionClassifier()),
            "params": {
                "classifier__threshold": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
                "classifier__fit_intercept": [True, False],
            },
        },
        "KNN": {
            "pipeline": build_model_pipeline(KNeighborsClassifier()),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 11, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        },
        "MLP": {
            "pipeline": build_model_pipeline(
                MLPClassifier(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    early_stopping=True,
                )
            ),
            "params": {
                "classifier__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                "classifier__activation": ["relu", "tanh"],
                "classifier__alpha": [0.0001, 0.001],
            },
        },
        "Naive_Bayes": {
            "pipeline": build_model_pipeline(GaussianNB()),
            "params": {
                "classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
        },
    }


def save_json(data: dict, path: Path) -> None:
    """
    Salva um dicionário em JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False, default=str)


def train_and_evaluate_models() -> None:
    """
    Treina os quatro modelos, compara as métricas e salva o melhor modelo.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_create_processed_dataset()

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models_config = get_models_and_parameters()

    results = {}

    print("Iniciando treinamento dos modelos...")
    print(f"Registros de treino: {x_train.shape[0]}")
    print(f"Registros de teste: {x_test.shape[0]}")

    for model_name, config in models_config.items():
        print(f"\nTreinando modelo: {model_name}")

        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            scoring="f1",
            cv=cross_validation,
            n_jobs=-1,
        )

        grid_search.fit(x_train, y_train)

        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(x_test)

        metrics = calculate_classification_metrics(y_test, y_pred)

        results[model_name] = {
            "best_params": grid_search.best_params_,
            "best_cv_f1_score": round(float(grid_search.best_score_), 4),
            "test_metrics": metrics,
        }

        print(f"Melhores parâmetros: {grid_search.best_params_}")
        print(f"F1 médio na validação cruzada: {round(float(grid_search.best_score_), 4)}")
        print_metrics(model_name, metrics)

    best_model_name = max(
        results,
        key=lambda name: results[name]["test_metrics"]["f1_score"],
    )

    best_model_config = models_config[best_model_name]

    print(f"\nRetreinando melhor modelo com todos os dados de treino: {best_model_name}")

    final_grid = GridSearchCV(
        estimator=best_model_config["pipeline"],
        param_grid=best_model_config["params"],
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
    )

    final_grid.fit(x_train, y_train)

    joblib.dump(final_grid.best_estimator_, BEST_MODEL_PATH)

    model_info = {
        "best_model_name": best_model_name,
        "best_params": final_grid.best_params_,
        "target_mapping": {
            "Movie": 0,
            "TV Show": 1,
        },
        "positive_class": "TV Show",
        "negative_class": "Movie",
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "selection_criterion": "Maior F1-score no conjunto de teste",
    }

    save_json(results, METRICS_PATH)
    save_json(model_info, MODEL_INFO_PATH)

    print("\nTreinamento finalizado.")
    print(f"Melhor modelo: {best_model_name}")
    print(f"Modelo salvo em: {BEST_MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(f"Informações do modelo salvas em: {MODEL_INFO_PATH}")


if __name__ == "__main__":
    train_and_evaluate_models()