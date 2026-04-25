#headline
import os
from functools import partial                      
import pandas as pd
from datasets import Dataset
from config import Config, get_config  
from sentence_transformers import SentenceTransformer            

# === Adaptation au schéma réel du dataset ===
# Colonnes attendues dans le CSV : intent, postContent, mauvaisPostContent1, mauvaisPostContent2
INTENT_COL = "intent"                                       # intention donnée
POSTCONTENT_COL = "postContent"                             # post "positif" associé 
NEGATIVE_PREFIX = "mauvaisPostContent_"                     # préfixe commun des colonnes négatives (mauvaispostcontent1, mauvaispostcontent2)

# Noms internes utilisés par le pipeline
ANCHOR_COL = "intent"
POSITIVE_COL = "postContent"
NEGATIVE_COLS = "mauvaisPostContent_{}"


def _preproc_fn(examples, n_negative):              
    
    """
    Prétraitement batched :
    Conserve les champs avec les mêmes clés que le dataset
    """
    
    negatives = {}
    for i in range(1, n_negative + 1):
        neg_col = NEGATIVE_COLS.format(i)
        negative = [str(x) for x in examples[neg_col]]
        negatives[neg_col] = negative

    return {**negatives}

#necessaires pour definir les trois méthodes suivantes

model=SentenceTransformer("intfloat/multilingual-e5-small")
tokenizer=model.tokenizer
max_len=512

def chunk_text(text,tokenizer=tokenizer,max_len=max_len):

    """

    chunk un texte lorsque le nombre de tokens est plus grand que 512
    
    """

    tokens=tokenizer.tokenize(str(text))
    chunks=[]

    for i in range(0, len(tokens), max_len):
        chunk = tokens[i:i+max_len]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks or [""]


def apply_chunking(df, tokenizer, max_len=512):

    """

    chunking appliqué sur le dataset si necessaire

    """


    for col in [ANCHOR_COL, POSITIVE_COL,NEGATIVE_COLS] :

        df[col] = df[col].apply(lambda x: chunk_text(x, tokenizer, max_len))

    return df


def explode_chunks(df):

    """   

    Transforme un df où certaines colonnes contiennent des listes de chunks
    en un df plat où chaque chunk devient une ligne et les autres colonnes sont répétées pour chaque chunk

    """

    # Colonnes à exploser
    cols_to_explode = [ANCHOR_COL, POSITIVE_COL,NEGATIVE_COLS]

    # Exploser chaque colonne de chunks
    for col in cols_to_explode:
        df = df.explode(col, ignore_index=True)

    return df



def get_train_dataset(args: Config):                                                
    
    """
    Charge ou construit le dataset d’entraînement compatible avec SentenceTransformer :
    - Lit le CSV (colonnes : intent, postContent, mauvaisPostContent1..k)
    - Valide la présence des colonnes
    - Renomme vers les noms internes (anchor, positive, negative1..k)
    - Met en cache le dataset sur disque
    """
    # Si un cache existe, on le réutilise directement pour le training
    if os.path.exists(args.cache_data_path):                                        

        print(f"Chargement du dataset depuis {args.cache_data_path}")
        dataset = Dataset.load_from_disk(args.cache_data_path)
        return dataset

    # Lecture du CSV (encodage robuste pour fichiers Windows/Excel)

    # --- LECTURE ROBUSTE DU CSV ---

    # gestion du séparataeur
    try:
        # Détection automatique du séparateur + gestion des guillemets
        data = pd.read_csv(
            args.data_path,           # chemin du fichier csv à lire
            encoding="latin-1",       # encodage utilisé pour ls accents
            sep=None,                 # autodétection du separateur par pandas
            engine="python",          # nécessaire pour sep=None/regex
            quotechar='"',            # caractere utilisé pour entourer du texte
            escapechar='\\'           # accepte \ dans les champ comme un caractere en soi
        )
    except Exception as e1:           # Si l'autodétection échoue, tenter le point-virgule
 
        try:
            data = pd.read_csv(
                args.data_path,
                encoding="latin-1",
                sep=';',              # tenter le pont virgule
                engine="python",
                quotechar='"',
                escapechar='\\'
            )
        except Exception as e2:

            data = pd.read_csv(
                args.data_path,
                encoding="latin-1",
                engine="python",
                sep=r',(?=(?:[^"]*"[^"]*")*[^"]*$)',  # virgules hors "..."     # Dernier recours : séparer par virgules hors guillemets (regex)
                quotechar='"',
                escapechar='\\'
            )

    # Option : sous-échantillonnage pour des tests rapides du csv
    if args.limit:
        data = data.sample(n=args.limit, random_state=args.seed)

    cols = data.columns.tolist()

    # Vérifications minimales des colonnes
    assert INTENT_COL in cols, f"{INTENT_COL} absent des colonnes : {cols}"                            
    assert POSTCONTENT_COL in cols, f"{POSTCONTENT_COL} absent des colonnes : {cols}"

    # Détection automatique des colonnes négatives (negative1, negative2, ...)
    negative_cols = [c for c in cols if c.startswith(NEGATIVE_PREFIX)]                                
    n_negative = len(negative_cols)
    assert n_negative > 0, (
        f"Aucune colonne négative détectée avec le préfixe '{NEGATIVE_PREFIX}' dans : {cols}"          
    )

    # Nettoyage simple : supprime les lignes entièrement vides et remplace NaN par chaîne vide
    data = data.dropna(how="all").fillna("")

    # Renommage vers les noms internes utilisés par le pipeline
    rename_map = {
        INTENT_COL: ANCHOR_COL,
        POSTCONTENT_COL: POSITIVE_COL,
    } | {col: NEGATIVE_COLS.format(i + 1) for i, col in enumerate(negative_cols)}
    data = data.rename(columns=rename_map)

    # Élimination des doublons (mêmes paires anchor/positive)
    data = data.drop_duplicates(subset=[ANCHOR_COL, POSITIVE_COL])

    # Mise en place des chunks(si > 512 tokens) 

    data=apply_chunking(data,tokenizer)

    # Explosion des listes en lignes indépendantes

    data=explode_chunks(data)

    # Construction du dataset HuggingFace
    dataset = Dataset.from_pandas(data).select_columns(
        [ANCHOR_COL, POSITIVE_COL] + [NEGATIVE_COLS.format(i + 1) for i in range(n_negative)]
    )


    # Mise en cache sur disque pour accélérer les runs suivants
    os.makedirs(args.cache_data_path, exist_ok=True)
    dataset.save_to_disk(args.cache_data_path)

    return dataset            


def get_val_dataset(args: Config):

    """À implémenter si une validation est nécessaire (même logique que train)."""
    pass


#  premiers tests
def test():
    """
    Test rapide :
    - Construit le dataset avec la config de test
    - Vérifie l’accès aux champs
    - Tokenize un exemple pour contrôle
    """
    from transformers import AutoTokenizer
    from config import test_config

    train_dataset = get_train_dataset(test_config)
    print(train_dataset)
    print(train_dataset[0]["anchor"])

    model = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="cache")         

    inputs = tokenizer("🚀", return_tensors="pt")
    print(inputs)
    print(inputs["input_ids"].shape)


if __name__ == "__main__":

    # Point d’entrée : récupère la configuration choisie via --config et affiche un aperçu
    config = get_config()
    train_dataset = get_train_dataset(config)
    print(train_dataset.select(range(10)).to_pandas().head())       
