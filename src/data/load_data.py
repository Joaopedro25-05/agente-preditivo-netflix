from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/netflix_titles.csv")


def load_netflix_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega o dataset Netflix Movies and TV Shows.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        DataFrame com os dados carregados.

    Raises:
        FileNotFoundError: Caso o arquivo CSV não exista no caminho esperado.
        ValueError: Caso o arquivo esteja vazio.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado em: {path}. "
            "Verifique se o arquivo netflix_titles.csv está dentro de data/raw/."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("O arquivo CSV foi carregado, mas está vazio.")

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Gera um resumo inicial do dataset.

    Args:
        df: DataFrame carregado.

    Returns:
        Dicionário com informações básicas da base.
    """
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    return summary


if __name__ == "__main__":
    dataset = load_netflix_dataset()
    summary = get_dataset_summary(dataset)

    print("Dataset carregado com sucesso.")
    print(f"Linhas: {summary['rows']}")
    print(f"Colunas: {summary['columns']}")
    print(f"Duplicatas: {summary['duplicated_rows']}")
    print("\nColunas encontradas:")
    for column in summary["column_names"]:
        print(f"- {column}")

    print("\nValores ausentes por coluna:")
    for column, missing_count in summary["missing_values"].items():
        print(f"- {column}: {missing_count}")