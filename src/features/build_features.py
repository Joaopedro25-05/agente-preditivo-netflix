from pathlib import Path

import pandas as pd

from src.data.load_data import load_netflix_dataset


PROCESSED_DATA_PATH = Path("data/processed/netflix_modeling.csv")


FEATURE_COLUMNS = [
    "release_year",
    "year_added",
    "month_added",
    "num_genres",
    "added_delay",
    "rating",
    "country_first",
]

TARGET_COLUMN = "target"


def fix_rating_duration_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige casos em que valores de duração aparecem indevidamente na coluna rating.

    Exemplo do problema:
    - rating = "74 min"
    - duration = vazio

    Nesse caso:
    - duration recebe "74 min"
    - rating passa a ser nulo
    """
    df = df.copy()

    rating_as_text = df["rating"].astype("string")

    rating_contains_duration = rating_as_text.str.contains(
        r"min|Season",
        case=False,
        na=False,
    )

    duration_is_missing = df["duration"].isna()

    mask = duration_is_missing & rating_contains_duration

    df.loc[mask, "duration"] = df.loc[mask, "rating"]
    df.loc[mask, "rating"] = pd.NA

    return df


def extract_first_country(country_value: object) -> str:
    """
    Extrai o primeiro país informado na coluna country.

    Quando o país está ausente, retorna 'Unknown'.
    """
    if pd.isna(country_value):
        return "Unknown"

    countries = str(country_value).split(",")
    first_country = countries[0].strip()

    if not first_country:
        return "Unknown"

    return first_country


def count_genres(listed_in_value: object) -> int:
    """
    Conta quantas categorias/gêneros existem na coluna listed_in.
    """
    if pd.isna(listed_in_value):
        return 0

    genres = [genre.strip() for genre in str(listed_in_value).split(",")]
    genres = [genre for genre in genres if genre]

    return len(genres)


def extract_duration_number(duration_value: object) -> float:
    """
    Extrai o número da coluna duration.

    Exemplos:
    - '90 min' vira 90
    - '2 Seasons' vira 2

    Essa variável será mantida para análise exploratória,
    mas não será usada no treino principal para evitar vazamento de informação.
    """
    if pd.isna(duration_value):
        return pd.NA

    extracted = pd.Series([str(duration_value)]).str.extract(r"(\d+)")[0].iloc[0]

    if pd.isna(extracted):
        return pd.NA

    return float(extracted)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a base final de modelagem a partir do dataset bruto da Netflix.

    A variável alvo será:
    - Movie = 0
    - TV Show = 1
    """
    df = df.copy()

    df = fix_rating_duration_inconsistencies(df)

    df["date_added_clean"] = pd.to_datetime(
        df["date_added"].astype("string").str.strip(),
        errors="coerce",
    )

    df["year_added"] = df["date_added_clean"].dt.year
    df["month_added"] = df["date_added_clean"].dt.month

    df["added_delay"] = df["year_added"] - df["release_year"]

    df["duration_num"] = df["duration"].apply(extract_duration_number)

    df["num_genres"] = df["listed_in"].apply(count_genres)

    df["country_first"] = df["country"].apply(extract_first_country)

    df["target"] = df["type"].map(
        {
            "Movie": 0,
            "TV Show": 1,
        }
    )

    modeling_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "type"]

    modeling_df = df[modeling_columns].copy()

    return modeling_df


def save_processed_dataset(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """
    Salva a base processada em CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    raw_df = load_netflix_dataset()
    processed_df = build_features(raw_df)

    save_processed_dataset(processed_df)

    print("Base processada com sucesso.")
    print(f"Arquivo salvo em: {PROCESSED_DATA_PATH}")
    print(f"Linhas: {processed_df.shape[0]}")
    print(f"Colunas: {processed_df.shape[1]}")

    print("\nColunas da base processada:")
    for column in processed_df.columns:
        print(f"- {column}")

    print("\nDistribuição da variável alvo:")
    print(processed_df["type"].value_counts())

    print("\nValores ausentes por coluna:")
    print(processed_df.isnull().sum())