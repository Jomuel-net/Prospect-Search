import os
from functools import partial
import pandas as pd
from datasets import Dataset
from config import Config, get_config

# Configuration des colonnes du CSV source
INTENT_COL = "headline"
POSTCONTENT_COL = "job"
NEGATIVE_PREFIX = "negative"

# Noms de colonnes internes pour le pipeline
ANCHOR_COL = "anchor"
POSITIVE_COL = "positive"
NEGATIVE_COLS = "negative{}"


def _preproc_fn(examples, anchor_prefix, positive_prefix, n_negative):
    """
    Applique les préfixes (ex: 'query:', 'passage:') aux colonnes textuelles.
    """
    anchor = [anchor_prefix + str(x) for x in examples[ANCHOR_COL]]
    positive = [positive_prefix + str(x) for x in examples[POSITIVE_COL]]

    negatives = {}
    for i in range(1, n_negative + 1):
        neg_col = NEGATIVE_COLS.format(i)
        negative = [str(x) for x in examples[neg_col]]
        negative = [positive_prefix + x if x != "" else "" for x in negative]
        negatives[neg_col] = negative

    return {ANCHOR_COL: anchor, POSITIVE_COL: positive, **negatives}


def get_train_dataset(args: Config):
    """
    Charge, nettoie et formate le dataset d'entraînement depuis un CSV.
    Gère le cache disque pour accélérer les exécutions futures.
    """
    if os.path.exists(args.cache_data_path):
        return Dataset.load_from_disk(args.cache_data_path)

    # Tentatives de lecture du CSV avec différents séparateurs pour robustesse
    try:
        data = pd.read_csv(
            args.data_path, encoding="latin-1", sep=None, engine="python",
            quotechar='"', escapechar='\\'
        )
    except Exception:
        try:
            data = pd.read_csv(
                args.data_path, encoding="latin-1", sep=';', engine="python",
                quotechar='"', escapechar='\\'
            )
        except Exception:
            # Fallback regex pour les cas complexes
            data = pd.read_csv(
                args.data_path, encoding="latin-1", engine="python",
                sep=r',(?=(?:[^"]*"[^"]*")*[^"]*$)', quotechar='"', escapechar='\\'
            )

    if args.limit:
        data = data.sample(n=args.limit, random_state=args.seed)

    cols = data.columns.tolist()

    # Validation de la structure du fichier
    assert INTENT_COL in cols, f"{INTENT_COL} manquant."
    assert POSTCONTENT_COL in cols, f"{POSTCONTENT_COL} manquant."

    negative_cols = [c for c in cols if c.startswith(NEGATIVE_PREFIX)]
    n_negative = len(negative_cols)
    assert n_negative > 0, "Aucune colonne négative trouvée."

    # Nettoyage et renommage
    data = data.dropna(how="all").fillna("")
    
    rename_map = {
        INTENT_COL: ANCHOR_COL,
        POSTCONTENT_COL: POSITIVE_COL,
    } | {col: NEGATIVE_COLS.format(i + 1) for i, col in enumerate(negative_cols)}
    
    data = data.rename(columns=rename_map)
    data = data.drop_duplicates(subset=[ANCHOR_COL, POSITIVE_COL])

    # Conversion en Dataset HuggingFace
    dataset = Dataset.from_pandas(data).select_columns(
        [ANCHOR_COL, POSITIVE_COL] + [NEGATIVE_COLS.format(i + 1) for i in range(n_negative)]
    )

    if args.needs_prefix:
        preproc_fn = partial(
            _preproc_fn,
            anchor_prefix=args.prefix[0],
            positive_prefix=args.prefix[1],
            n_negative=n_negative,
        )
        dataset = dataset.map(preproc_fn, batched=True, desc="Prétraitement")

    os.makedirs(args.cache_data_path, exist_ok=True)
    dataset.save_to_disk(args.cache_data_path)

    return dataset


def get_val_dataset(args: Config):
    pass


def test():
    from transformers import AutoTokenizer
    from config import test_config

    train_dataset = get_train_dataset(test_config)
    print(f"Dataset size: {len(train_dataset)}")
    
    model = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="cache")
    inputs = tokenizer("🚀", return_tensors="pt")
    print(f"Tokenized shape: {inputs['input_ids'].shape}")


if __name__ == "__main__":
    config = get_config()
    train_dataset = get_train_dataset(config)
    print(train_dataset.select(range(min(10, len(train_dataset)))).to_pandas().head())