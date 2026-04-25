# implementation de config pour main ( permet de choisir dynamiquement un profil d'entrainement)
import argparse                                                 
from dataclasses import dataclass                               
from pathlib import Path                                       
from dotenv import load_dotenv                                  
import os                                                       
from datetime import datetime                                 

# ===============================
# Détection automatique du dossier src/
# ===============================

# chemin où se trouve le dossier src en local
def get_src_root() -> Path:
    """
    Retourne le chemin absolu du dossier src/ (où se trouve ce fichier config.py)
    """
    return Path(__file__).resolve().parent

SRC_ROOT = get_src_root()                                       

# Charge les variables depuis .env (si présent dans src/)
load_dotenv(SRC_ROOT / ".env")

# ===============================
# Génération du nom de dossier unique
# ===============================
def get_output_dir_with_timestamp() -> str:
    """
    Génère un nom de dossier unique avec timestamp
    Format: output_finetuned_model_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")            
    return str(SRC_ROOT / f"output_finetuned_model_{timestamp}")    

# ===============================
# Configuration générale du projet
# ===============================

@dataclass(frozen=True)          # rend la classe immuable une fois crée

class Config:
    # === Modèle ===
    model_id: str = ""                                               # identifiant du modèle sur Hugging Face
    prefix: tuple = ("query: ", "passage: ")                         # préfixes utilisés dans le texte (requête / passage) [utile pour headline uniquement]

    # === Données (chemins RELATIFS à src/) ===
    data_path: str = os.getenv("DATA_PATH", str(SRC_ROOT / "content_cleaned_data.csv"))      
    seed: int = 42
    limit: int = None
    num_workers: int = 8                  
    prefetch_factor: int = 2                
    cache_data_path: str = os.getenv("CACHE_DIR", str(SRC_ROOT / "cache" / "cached_data"))  
    needs_prefix: bool = True                # indique si on doit ajouter les préfixes query/passage

    # === Hyper-Parametres Entraînement ===
    batch_size: int = 16     
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    decay_ratio: float = 0.1

    # === Hyper-Parametres Évaluation (à implémenter si besoin) ===
    eval_steps: int = 1000
    evaluation_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    logging_steps: int = 1000

    # === Sauvegarde (avec timestamp unique) ===
    output_dir: str = os.getenv("OUTPUT_DIR", get_output_dir_with_timestamp()) 
    
    def __post_init__(self):

        # Vérification : le préfixe doit être un tuple de 2 éléments
        assert len(self.prefix) == 2, "le prefixe doit etre un tuple de longueur 2"


# ===============================
# Configurations spécifiques
# ===============================

# Petite configuration pour test rapide
test_config = Config(
    model_id="intfloat/multilingual-e5-small",
    batch_size=16,
    limit=100,
    needs_prefix=False
)
# Entraînement standard (taille réduite)
small_config = Config(
    model_id="intfloat/multilingual-e5-small",
    batch_size=32,
)

# Entraînement de base (modèle intermédiaire)
base_config = Config(
    model_id="intfloat/multilingual-e5-base",
    batch_size=16,
)

# Entraînement plus lourd (modèle large)
large_config = Config(
    model_id="intfloat/multilingual-e5-large",
    batch_size=8,
)

# Configuration pour données de contenu spécifiques
content_config = Config(
    model_id="intfloat/multilingual-e5-small",
    needs_prefix=False,                 # pas de query / passage pour content
)


# ===============================
# Sélecteur de configuration
# ===============================
def get_config() -> Config:

    """
    Sélectionne la configuration à utiliser en ligne de commande :
    ex.  python main.py --config small

    """

    parser = argparse.ArgumentParser(description="Train a model")           
    parser.add_argument(
        "--config",
        type=str,
        default="content",                                                     
        choices=["test", "small", "base", "large", "content"],
    )   

    args = parser.parse_args() 

    CONFS = {                  
        "test": test_config,
        "small": small_config,
        "base": base_config,
        "large": large_config,
        "content": content_config,
    }

    config = CONFS.get(args.config)          

    if config is None:   
        raise ValueError(
            f"Unknown config: {args.config}. Choisir parmi {', '.join(CONFS.keys())}."
        )

    return config                          

    
    
   