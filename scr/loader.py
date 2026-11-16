import os
from pathlib import Path
import pandas as pd


def carregar_fake_br(
    base_path="data/raw/Fake.br-Corpus/size_normalized_texts",
):
    dados = []
    base_dir = Path(base_path)

    # Mapeamento dos rótulos (1 para fake, 0 para verdadeira)
    categorias = {"fake": 1, "true": 0}

    for categoria, label in categorias.items():
        pasta = base_dir / categoria
        if not pasta.exists():
            print(f"Aviso: Pasta {pasta} não encontrada.")
            continue

        for arquivo in pasta.glob("*.txt"):
            with open(arquivo, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read().strip()
                if texto:
                    dados.append(
                        {
                            "id_original": arquivo.stem,
                            "texto": texto,
                            "label": label,
                        }
                    )

    df = pd.DataFrame(dados)
    print(
        f"Total de notícias carregadas: {len(df)} ({df['label'].value_counts().to_dict()})"
    )
    return df


def criar_amostra_equilibrada(df, n_amostras=500, output_path="data/processed/dataset_amostra.csv"):
    # Extrai n_amostras divididas igualmente entre fake (1) e verdadeira (0)
    n_por_classe = n_amostras // 2
    df_fake = df[df["label"] == 1].sample(n=n_por_classe, random_state=42)
    df_true = df[df["label"] == 0].sample(n=n_por_classe, random_state=42)

    df_amostra = (
        pd.concat([df_fake, df_true])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Salva em data/processed/ para acesso rápido nas simulações evolutivas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_amostra.to_csv(output_path, index=False)
    print(f"Amostra salva com sucesso em '{output_path}' com {len(df_amostra)} notícias.")
    return df_amostra