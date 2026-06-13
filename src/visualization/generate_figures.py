import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.load_data import load_netflix_dataset
from src.features.build_features import build_features


FIGURES_DIR = Path("reports/figures")
METRICS_PATH = Path("artifacts/metrics.json")


def save_current_figure(filename: str) -> None:
    """
    Salva a figura atual na pasta reports/figures.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura salva: {output_path}")


def extract_duration_number(duration_value: object) -> float | None:
    """
    Extrai o valor numérico da coluna duration.

    Exemplos:
    - '90 min' -> 90
    - '2 Seasons' -> 2
    """
    if pd.isna(duration_value):
        return None

    match = re.search(r"(\d+)", str(duration_value))

    if not match:
        return None

    return float(match.group(1))


def prepare_exploratory_dataframe() -> pd.DataFrame:
    """
    Prepara uma base auxiliar para análise exploratória.
    """
    raw_df = load_netflix_dataset()
    processed_df = build_features(raw_df)

    exploratory_df = raw_df.copy()

    exploratory_df["target"] = processed_df["target"]
    exploratory_df["year_added"] = processed_df["year_added"]
    exploratory_df["month_added"] = processed_df["month_added"]
    exploratory_df["added_delay"] = processed_df["added_delay"]
    exploratory_df["num_genres"] = processed_df["num_genres"]
    exploratory_df["country_first"] = processed_df["country_first"]
    exploratory_df["duration_num"] = exploratory_df["duration"].apply(extract_duration_number)

    return exploratory_df


def generate_type_frequency_chart(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="type", order=df["type"].value_counts().index)
    plt.title("Frequência por tipo de conteúdo")
    plt.xlabel("Tipo")
    plt.ylabel("Quantidade")
    save_current_figure("frequencia_tipo_conteudo.png")


def generate_rating_frequency_chart(df: pd.DataFrame) -> None:
    top_ratings = df["rating"].fillna("Unknown").value_counts().head(10).index

    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df[df["rating"].fillna("Unknown").isin(top_ratings)],
        x=df["rating"].fillna("Unknown"),
        order=top_ratings,
    )
    plt.title("Top 10 classificações indicativas mais frequentes")
    plt.xlabel("Classificação indicativa")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=45)
    save_current_figure("frequencia_classificacao_indicativa.png")


def generate_top_countries_chart(df: pd.DataFrame) -> None:
    top_countries = df["country_first"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=top_countries.index,
        y=top_countries.values,
    )
    plt.title("Top 10 países principais mais frequentes")
    plt.xlabel("País principal")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=45)
    save_current_figure("frequencia_top_10_paises.png")


def generate_correlation_heatmap(df: pd.DataFrame) -> None:
    correlation_columns = [
        "release_year",
        "year_added",
        "month_added",
        "duration_num",
        "num_genres",
        "added_delay",
        "target",
    ]

    correlation_df = df[correlation_columns].copy()
    correlation_matrix = correlation_df.corr(numeric_only=True)

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )
    plt.title("Matriz de correlação das variáveis numéricas")
    save_current_figure("matriz_correlacao.png")


def generate_release_year_boxplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="type", y="release_year")
    plt.title("Distribuição do ano de lançamento por tipo")
    plt.xlabel("Tipo")
    plt.ylabel("Ano de lançamento")
    save_current_figure("boxplot_ano_lancamento_por_tipo.png")


def generate_added_delay_boxplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="type", y="added_delay")
    plt.title("Distribuição do atraso de adição por tipo")
    plt.xlabel("Tipo")
    plt.ylabel("Diferença entre ano de adição e lançamento")
    save_current_figure("boxplot_added_delay_por_tipo.png")


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Arquivo artifacts/metrics.json não encontrado. "
            "Execute primeiro: python -m src.models.train_models"
        )

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_metrics_comparison_chart(metrics: dict) -> None:
    rows = []

    for model_name, model_result in metrics.items():
        test_metrics = model_result["test_metrics"]

        rows.append(
            {
                "Modelo": model_name,
                "Acurácia": test_metrics["accuracy"],
                "Sensibilidade": test_metrics["sensitivity"],
                "Especificidade": test_metrics["specificity"],
                "Precisão": test_metrics["precision"],
                "F1-score": test_metrics["f1_score"],
            }
        )

    metrics_df = pd.DataFrame(rows)

    melted_df = metrics_df.melt(
        id_vars="Modelo",
        var_name="Métrica",
        value_name="Valor",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_df, x="Modelo", y="Valor", hue="Métrica")
    plt.title("Comparação das métricas dos modelos")
    plt.xlabel("Modelo")
    plt.ylabel("Valor")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    save_current_figure("comparacao_metricas_modelos.png")


def generate_confusion_matrix_charts(metrics: dict) -> None:
    for model_name, model_result in metrics.items():
        confusion = model_result["test_metrics"]["confusion_matrix"]

        matrix = [
            [confusion["tn"], confusion["fp"]],
            [confusion["fn"], confusion["tp"]],
        ]

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predito Movie", "Predito TV Show"],
            yticklabels=["Real Movie", "Real TV Show"],
        )
        plt.title(f"Matriz de confusão - {model_name}")
        plt.xlabel("Classe prevista")
        plt.ylabel("Classe real")

        safe_model_name = model_name.lower().replace(" ", "_").replace("ã", "a")
        save_current_figure(f"matriz_confusao_{safe_model_name}.png")


def generate_all_figures() -> None:
    sns.set_theme(style="whitegrid")

    df = prepare_exploratory_dataframe()
    metrics = load_metrics()

    generate_type_frequency_chart(df)
    generate_rating_frequency_chart(df)
    generate_top_countries_chart(df)
    generate_correlation_heatmap(df)
    generate_release_year_boxplot(df)
    generate_added_delay_boxplot(df)
    generate_metrics_comparison_chart(metrics)
    generate_confusion_matrix_charts(metrics)

    print("\nTodas as figuras foram geradas com sucesso.")


if __name__ == "__main__":
    generate_all_figures()